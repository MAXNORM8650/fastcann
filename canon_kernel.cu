#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

namespace {

constexpr int THREADS = 256;
constexpr int TILE_L = 64;
constexpr int CTILE = 4;
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_CTILE = 4;

__global__ void canon_fwd_fp32_kernel_k4(
    const float* __restrict__ x,
    const float* __restrict__ mix,
    float* __restrict__ y,
    int B,
    int L,
    int D) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tile_l0 = blockIdx.x * TILE_L;
    const int c0 = blockIdx.y * CTILE;
    const int b = blockIdx.z;

    const int t = tile_l0 + tx;
    const int c = c0 + ty;

    __shared__ float smem[CTILE][TILE_L + 3];

    for (int s = tx; s < TILE_L + 3; s += blockDim.x) {
        const int g_t = tile_l0 - 1 + s;
        float v = 0.0f;
        if (c < D && g_t >= 0 && g_t < L) {
            v = x[(b * L + g_t) * D + c];
        }
        smem[ty][s] = v;
    }
    __syncthreads();

    if (t < L && c < D) {
        const float w0 = mix[0 * D + c];
        const float w1 = mix[1 * D + c];
        const float w2 = mix[2 * D + c];
        const float w3 = mix[3 * D + c];

        const float xm1 = smem[ty][tx + 0];
        const float x0 = smem[ty][tx + 1];
        const float xp1 = smem[ty][tx + 2];
        const float xp2 = smem[ty][tx + 3];

        y[(b * L + t) * D + c] = x0 + w0 * xp2 + w1 * xp1 + w2 * x0 + w3 * xm1;
    }
}

__global__ void canon_bwd_dx_fp32_kernel_k4(
    const float* __restrict__ grad_out,
    const float* __restrict__ mix,
    float* __restrict__ grad_x,
    int B,
    int L,
    int D) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tile_l0 = blockIdx.x * TILE_L;
    const int c0 = blockIdx.y * CTILE;
    const int b = blockIdx.z;

    const int t = tile_l0 + tx;
    const int c = c0 + ty;

    __shared__ float smem[CTILE][TILE_L + 3];

    for (int s = tx; s < TILE_L + 3; s += blockDim.x) {
        const int g_t = tile_l0 - 2 + s;
        float v = 0.0f;
        if (c < D && g_t >= 0 && g_t < L) {
            v = grad_out[(b * L + g_t) * D + c];
        }
        smem[ty][s] = v;
    }
    __syncthreads();

    if (t < L && c < D) {
        const float w0 = mix[0 * D + c];
        const float w1 = mix[1 * D + c];
        const float w2 = mix[2 * D + c];
        const float w3 = mix[3 * D + c];

        const float gm2 = smem[ty][tx + 0];
        const float gm1 = smem[ty][tx + 1];
        const float g0 = smem[ty][tx + 2];
        const float gp1 = smem[ty][tx + 3];

        grad_x[(b * L + t) * D + c] = g0 + w0 * gm2 + w1 * gm1 + w2 * g0 + w3 * gp1;
    }
}

__global__ void canon_bwd_dmix_fp32_kernel_k4(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    float* __restrict__ grad_mix,
    int B,
    int L,
    int D) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c0 = blockIdx.x * REDUCE_CTILE;
    const int c = c0 + ty;

    __shared__ float s0[REDUCE_CTILE][REDUCE_THREADS];
    __shared__ float s1[REDUCE_CTILE][REDUCE_THREADS];
    __shared__ float s2[REDUCE_CTILE][REDUCE_THREADS];
    __shared__ float s3[REDUCE_CTILE][REDUCE_THREADS];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    const int BL = B * L;

    for (int idx = tx; idx < BL; idx += REDUCE_THREADS) {
        if (c >= D) {
            break;
        }
        const int b = idx / L;
        const int t = idx - b * L;
        const int base = (b * L + t) * D + c;
        const float go = grad_out[base];

        if (t + 2 < L) acc0 += go * x[(b * L + (t + 2)) * D + c];
        if (t + 1 < L) acc1 += go * x[(b * L + (t + 1)) * D + c];
        acc2 += go * x[base];
        if (t - 1 >= 0) acc3 += go * x[(b * L + (t - 1)) * D + c];
    }

    s0[ty][tx] = acc0;
    s1[ty][tx] = acc1;
    s2[ty][tx] = acc2;
    s3[ty][tx] = acc3;
    __syncthreads();

    for (int stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            s0[ty][tx] += s0[ty][tx + stride];
            s1[ty][tx] += s1[ty][tx + stride];
            s2[ty][tx] += s2[ty][tx + stride];
            s3[ty][tx] += s3[ty][tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0 && c < D) {
        grad_mix[0 * D + c] = s0[ty][0];
        grad_mix[1 * D + c] = s1[ty][0];
        grad_mix[2 * D + c] = s2[ty][0];
        grad_mix[3 * D + c] = s3[ty][0];
    }
}

__global__ void canon_fwd_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mix,
    float* __restrict__ y,
    int B,
    int L,
    int D,
    int K,
    int center) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * L * D;
    if (idx >= total) {
        return;
    }

    const int c = idx % D;
    const int t = (idx / D) % L;
    const int b = idx / (L * D);

    float out = x[idx];
    for (int k = 0; k < K; ++k) {
        const int src_t = t + center - k;
        if (src_t >= 0 && src_t < L) {
            out += mix[k * D + c] * x[(b * L + src_t) * D + c];
        }
    }

    y[idx] = out;
}

__global__ void canon_bwd_dx_fp32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ mix,
    float* __restrict__ grad_x,
    int B,
    int L,
    int D,
    int K,
    int center) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * L * D;
    if (idx >= total) {
        return;
    }

    const int c = idx % D;
    const int t = (idx / D) % L;
    const int b = idx / (L * D);

    float out = grad_out[idx];
    for (int k = 0; k < K; ++k) {
        const int grad_t = t - center + k;
        if (grad_t >= 0 && grad_t < L) {
            out += mix[k * D + c] * grad_out[(b * L + grad_t) * D + c];
        }
    }

    grad_x[idx] = out;
}

__global__ void canon_bwd_dmix_fp32_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    float* __restrict__ grad_mix,
    int B,
    int L,
    int D,
    int K,
    int center) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (c >= D || k >= K) {
        return;
    }

    float acc = 0.0f;
    const int BL = B * L;
    for (int idx = 0; idx < BL; ++idx) {
        const int b = idx / L;
        const int t = idx - b * L;
        const int src_t = t + center - k;
        if (src_t >= 0 && src_t < L) {
            const int grad_base = (b * L + t) * D + c;
            const int x_base = (b * L + src_t) * D + c;
            acc += grad_out[grad_base] * x[x_base];
        }
    }

    grad_mix[k * D + c] = acc;
}

void check_inputs(torch::Tensor x, torch::Tensor mix) {
    CHECK_CUDA(x);
    CHECK_CUDA(mix);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(mix);
    CHECK_FLOAT(x);
    CHECK_FLOAT(mix);
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

    if (K == 4) {
        dim3 block(TILE_L, CTILE);
        dim3 grid((L + TILE_L - 1) / TILE_L,
                  (D + CTILE - 1) / CTILE,
                  B);

        canon_fwd_fp32_kernel_k4<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(),
            mix.data_ptr<float>(),
            y.data_ptr<float>(),
            B, L, D);
    } else {
        const int total = B * L * D;
        dim3 block(THREADS);
        dim3 grid((total + THREADS - 1) / THREADS);

        canon_fwd_fp32_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(),
            mix.data_ptr<float>(),
            y.data_ptr<float>(),
            B, L, D, K, center);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

std::vector<torch::Tensor> canon_backward_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor mix) {
    check_inputs(grad_out, mix);
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 3, "x must be [B, L, D]");
    TORCH_CHECK(x.sizes() == grad_out.sizes(), "x and grad_out must have the same shape");

    const int B = x.size(0);
    const int L = x.size(1);
    const int D = x.size(2);
    const int K = mix.size(0);
    const int center = K / 2;

    auto grad_x = torch::empty_like(x);
    auto grad_mix = torch::zeros({K, D}, x.options());

    if (K == 4) {
        dim3 block(TILE_L, CTILE);
        dim3 grid((L + TILE_L - 1) / TILE_L,
                  (D + CTILE - 1) / CTILE,
                  B);

        canon_bwd_dx_fp32_kernel_k4<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
            grad_out.data_ptr<float>(),
            mix.data_ptr<float>(),
            grad_x.data_ptr<float>(),
            B, L, D);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dim3 block_reduce(REDUCE_THREADS, REDUCE_CTILE);
        dim3 grid_reduce((D + REDUCE_CTILE - 1) / REDUCE_CTILE);

        canon_bwd_dmix_fp32_kernel_k4<<<grid_reduce, block_reduce, 0, at::cuda::getDefaultCUDAStream()>>>(
            grad_out.data_ptr<float>(),
            x.data_ptr<float>(),
            grad_mix.data_ptr<float>(),
            B, L, D);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        const int total = B * L * D;
        dim3 block(THREADS);
        dim3 grid((total + THREADS - 1) / THREADS);

        canon_bwd_dx_fp32_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
            grad_out.data_ptr<float>(),
            mix.data_ptr<float>(),
            grad_x.data_ptr<float>(),
            B, L, D, K, center);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dim3 block_reduce(THREADS);
        dim3 grid_reduce((D + THREADS - 1) / THREADS, K);

        canon_bwd_dmix_fp32_kernel<<<grid_reduce, block_reduce, 0, at::cuda::getDefaultCUDAStream()>>>(
            grad_out.data_ptr<float>(),
            x.data_ptr<float>(),
            grad_mix.data_ptr<float>(),
            B, L, D, K, center);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {grad_x, grad_mix};
}
