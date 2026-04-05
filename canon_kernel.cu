#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x) TORCH_CHECK( \
    x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, \
    #x " must be float32, float16, or bfloat16")

namespace {

constexpr int THREADS = 256;
constexpr int TILE_L = 64;
constexpr int CTILE = 4;
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_CTILE = 4;
constexpr int K2_TILE_L = 128;
constexpr int K2_CTILE = 2;
constexpr int K2_REDUCE_THREADS = 64;
constexpr int K2_REDUCE_CTILE = 8;

// Canon semantics:
// center = K / 2
// y[t] = x[t] + sum_k mix[k] * x[t + center - k]

template <int K, int TILE_L_, int CTILE_, typename scalar_t>
__global__ void canon_fwd_smallk_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ y,
    int B,
    int L,
    int D) {
    using acc_t = float;
    constexpr int CENTER = K / 2;
    constexpr int LEFT = K - 1 - CENTER;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_l0 = blockIdx.x * TILE_L_;
    const int c0 = blockIdx.y * CTILE_;
    const int b = blockIdx.z;
    const int t = tile_l0 + tx;
    const int c = c0 + ty;

    __shared__ acc_t smem[CTILE_][TILE_L_ + K - 1];

    for (int s = tx; s < TILE_L_ + K - 1; s += blockDim.x) {
        const int g_t = tile_l0 - LEFT + s;
        acc_t v = 0.0f;
        if (c < D && g_t >= 0 && g_t < L) {
            v = static_cast<acc_t>(x[(b * L + g_t) * D + c]);
        }
        smem[ty][s] = v;
    }
    __syncthreads();

    if (t < L && c < D) {
        acc_t acc = smem[ty][tx + LEFT];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            acc += static_cast<acc_t>(mix[k * D + c]) * smem[ty][tx + (K - 1 - k)];
        }
        y[(b * L + t) * D + c] = static_cast<scalar_t>(acc);
    }
}

template <int K, int TILE_L_, int CTILE_, typename scalar_t>
__global__ void canon_bwd_dx_smallk_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ grad_x,
    int B,
    int L,
    int D) {
    using acc_t = float;
    constexpr int CENTER = K / 2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_l0 = blockIdx.x * TILE_L_;
    const int c0 = blockIdx.y * CTILE_;
    const int b = blockIdx.z;
    const int t = tile_l0 + tx;
    const int c = c0 + ty;

    __shared__ acc_t smem[CTILE_][TILE_L_ + K - 1];

    for (int s = tx; s < TILE_L_ + K - 1; s += blockDim.x) {
        const int g_t = tile_l0 - CENTER + s;
        acc_t v = 0.0f;
        if (c < D && g_t >= 0 && g_t < L) {
            v = static_cast<acc_t>(grad_out[(b * L + g_t) * D + c]);
        }
        smem[ty][s] = v;
    }
    __syncthreads();

    if (t < L && c < D) {
        acc_t acc = smem[ty][tx + CENTER];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            acc += static_cast<acc_t>(mix[k * D + c]) * smem[ty][tx + k];
        }
        grad_x[(b * L + t) * D + c] = static_cast<scalar_t>(acc);
    }
}

template <int K, int REDUCE_THREADS_, int REDUCE_CTILE_, typename scalar_t>
__global__ void canon_bwd_dmix_smallk_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_mix,
    int B,
    int L,
    int D) {
    using acc_t = float;
    constexpr int CENTER = K / 2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c0 = blockIdx.x * REDUCE_CTILE_;
    const int c = c0 + ty;

    __shared__ acc_t shared[K][REDUCE_CTILE_][REDUCE_THREADS_];
    acc_t acc[K];

    #pragma unroll
    for (int k = 0; k < K; ++k) {
        acc[k] = 0.0f;
    }

    const int BL = B * L;
    for (int idx = tx; idx < BL; idx += REDUCE_THREADS_) {
        if (c >= D) {
            break;
        }
        const int b = idx / L;
        const int t = idx - b * L;
        const int base = (b * L + t) * D + c;
        const acc_t go = static_cast<acc_t>(grad_out[base]);

        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const int src_t = t + CENTER - k;
            if (src_t >= 0 && src_t < L) {
                acc[k] += go * static_cast<acc_t>(x[(b * L + src_t) * D + c]);
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < K; ++k) {
        shared[k][ty][tx] = acc[k];
    }
    __syncthreads();

    for (int stride = REDUCE_THREADS_ / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                shared[k][ty][tx] += shared[k][ty][tx + stride];
            }
        }
        __syncthreads();
    }

    if (tx == 0 && c < D) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            grad_mix[k * D + c] = static_cast<scalar_t>(shared[k][ty][0]);
        }
    }
}

template <typename scalar_t>
__global__ void canon_fwd_generic_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ y,
    int B,
    int L,
    int D,
    int K,
    int center) {
    using acc_t = float;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * L * D;
    if (idx >= total) {
        return;
    }

    const int c = idx % D;
    const int t = (idx / D) % L;
    const int b = idx / (L * D);

    acc_t out = static_cast<acc_t>(x[idx]);
    for (int k = 0; k < K; ++k) {
        const int src_t = t + center - k;
        if (src_t >= 0 && src_t < L) {
            out += static_cast<acc_t>(mix[k * D + c]) * static_cast<acc_t>(x[(b * L + src_t) * D + c]);
        }
    }

    y[idx] = static_cast<scalar_t>(out);
}

template <typename scalar_t>
__global__ void canon_bwd_dx_generic_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ mix,
    scalar_t* __restrict__ grad_x,
    int B,
    int L,
    int D,
    int K,
    int center) {
    using acc_t = float;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * L * D;
    if (idx >= total) {
        return;
    }

    const int c = idx % D;
    const int t = (idx / D) % L;
    const int b = idx / (L * D);

    acc_t out = static_cast<acc_t>(grad_out[idx]);
    for (int k = 0; k < K; ++k) {
        const int grad_t = t - center + k;
        if (grad_t >= 0 && grad_t < L) {
            out += static_cast<acc_t>(mix[k * D + c]) * static_cast<acc_t>(grad_out[(b * L + grad_t) * D + c]);
        }
    }

    grad_x[idx] = static_cast<scalar_t>(out);
}

template <typename scalar_t>
__global__ void canon_bwd_dmix_generic_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_mix,
    int B,
    int L,
    int D,
    int K,
    int center) {
    using acc_t = float;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (c >= D || k >= K) {
        return;
    }

    acc_t acc = 0.0f;
    const int BL = B * L;
    for (int idx = 0; idx < BL; ++idx) {
        const int b = idx / L;
        const int t = idx - b * L;
        const int src_t = t + center - k;
        if (src_t >= 0 && src_t < L) {
            const int grad_base = (b * L + t) * D + c;
            const int x_base = (b * L + src_t) * D + c;
            acc += static_cast<acc_t>(grad_out[grad_base]) * static_cast<acc_t>(x[x_base]);
        }
    }

    grad_mix[k * D + c] = static_cast<scalar_t>(acc);
}

void check_inputs(torch::Tensor x, torch::Tensor mix) {
    CHECK_CUDA(x);
    CHECK_CUDA(mix);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(mix);
    CHECK_DTYPE(x);
    CHECK_DTYPE(mix);
    TORCH_CHECK(x.scalar_type() == mix.scalar_type(), "x and mix must have the same dtype");
    TORCH_CHECK(x.dim() == 3, "x must be [B, L, D]");
    TORCH_CHECK(mix.dim() == 2, "mix must be [K, D]");
    TORCH_CHECK(mix.size(0) > 0, "mix must have shape [K, D] with K > 0");
    TORCH_CHECK(x.size(2) == mix.size(1), "D mismatch between x and mix");
}

} // namespace

torch::Tensor canon_forward_cuda(torch::Tensor x, torch::Tensor mix) {
    check_inputs(x, mix);

    const int B = x.size(0);
    const int L = x.size(1);
    const int D = x.size(2);
    const int K = mix.size(0);
    const int center = K / 2;

    auto y = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "canon_forward_cuda", [&] {
        if (K == 2) {
            dim3 block(K2_TILE_L, K2_CTILE);
            dim3 grid((L + K2_TILE_L - 1) / K2_TILE_L,
                      (D + K2_CTILE - 1) / K2_CTILE,
                      B);

            canon_fwd_smallk_kernel<2, K2_TILE_L, K2_CTILE, scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                x.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                B, L, D);
        } else if (K == 3) {
            dim3 block(TILE_L, CTILE);
            dim3 grid((L + TILE_L - 1) / TILE_L,
                      (D + CTILE - 1) / CTILE,
                      B);

            canon_fwd_smallk_kernel<3, TILE_L, CTILE, scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                x.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                B, L, D);
        } else if (K == 4) {
            dim3 block(TILE_L, CTILE);
            dim3 grid((L + TILE_L - 1) / TILE_L,
                      (D + CTILE - 1) / CTILE,
                      B);

            canon_fwd_smallk_kernel<4, TILE_L, CTILE, scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                x.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                B, L, D);
        } else {
            const int total = B * L * D;
            dim3 block(THREADS);
            dim3 grid((total + THREADS - 1) / THREADS);

            canon_fwd_generic_kernel<scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                x.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                B, L, D, K, center);
        }
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<torch::Tensor> canon_backward_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor mix) {
    check_inputs(grad_out, mix);
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_DTYPE(x);
    TORCH_CHECK(x.dim() == 3, "x must be [B, L, D]");
    TORCH_CHECK(x.sizes() == grad_out.sizes(), "x and grad_out must have the same shape");

    const int B = x.size(0);
    const int L = x.size(1);
    const int D = x.size(2);
    const int K = mix.size(0);
    const int center = K / 2;

    auto grad_x = torch::empty_like(x);
    auto grad_mix = torch::zeros({K, D}, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "canon_backward_cuda", [&] {
        if (K == 2) {
            dim3 block(K2_TILE_L, K2_CTILE);
            dim3 grid((L + K2_TILE_L - 1) / K2_TILE_L,
                      (D + K2_CTILE - 1) / K2_CTILE,
                      B);

            canon_bwd_dx_smallk_kernel<2, K2_TILE_L, K2_CTILE, scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                B, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            dim3 block_reduce(K2_REDUCE_THREADS, K2_REDUCE_CTILE);
            dim3 grid_reduce((D + K2_REDUCE_CTILE - 1) / K2_REDUCE_CTILE);

            canon_bwd_dmix_smallk_kernel<2, K2_REDUCE_THREADS, K2_REDUCE_CTILE, scalar_t><<<grid_reduce, block_reduce, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                grad_mix.data_ptr<scalar_t>(),
                B, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (K == 3) {
            dim3 block(TILE_L, CTILE);
            dim3 grid((L + TILE_L - 1) / TILE_L,
                      (D + CTILE - 1) / CTILE,
                      B);

            canon_bwd_dx_smallk_kernel<3, TILE_L, CTILE, scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                B, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            dim3 block_reduce(REDUCE_THREADS, REDUCE_CTILE);
            dim3 grid_reduce((D + REDUCE_CTILE - 1) / REDUCE_CTILE);

            canon_bwd_dmix_smallk_kernel<3, REDUCE_THREADS, REDUCE_CTILE, scalar_t><<<grid_reduce, block_reduce, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                grad_mix.data_ptr<scalar_t>(),
                B, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (K == 4) {
            dim3 block(TILE_L, CTILE);
            dim3 grid((L + TILE_L - 1) / TILE_L,
                      (D + CTILE - 1) / CTILE,
                      B);

            canon_bwd_dx_smallk_kernel<4, TILE_L, CTILE, scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                B, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            dim3 block_reduce(REDUCE_THREADS, REDUCE_CTILE);
            dim3 grid_reduce((D + REDUCE_CTILE - 1) / REDUCE_CTILE);

            canon_bwd_dmix_smallk_kernel<4, REDUCE_THREADS, REDUCE_CTILE, scalar_t><<<grid_reduce, block_reduce, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                grad_mix.data_ptr<scalar_t>(),
                B, L, D);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            const int total = B * L * D;
            dim3 block(THREADS);
            dim3 grid((total + THREADS - 1) / THREADS);

            canon_bwd_dx_generic_kernel<scalar_t><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                mix.data_ptr<scalar_t>(),
                grad_x.data_ptr<scalar_t>(),
                B, L, D, K, center);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            dim3 block_reduce(THREADS);
            dim3 grid_reduce((D + THREADS - 1) / THREADS, K);

            canon_bwd_dmix_generic_kernel<scalar_t><<<grid_reduce, block_reduce, 0, at::cuda::getDefaultCUDAStream()>>>(
                grad_out.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                grad_mix.data_ptr<scalar_t>(),
                B, L, D, K, center);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });

    return {grad_x, grad_mix};
}
