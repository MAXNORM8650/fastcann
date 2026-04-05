import time
import torch
from canon import CanonLayerCUDA


class CanonLayerSlow(torch.nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.mix = torch.nn.Parameter(torch.empty(kernel_size, dim, dtype=torch.float32))
        torch.nn.init.normal_(self.mix, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        center = self.kernel_size // 2
        y = torch.zeros_like(x)
        for idx in range(self.kernel_size):
            offset = idx - center
            shifted = torch.zeros_like(x)
            if offset < 0:
                k = -offset
                shifted[:, : seqlen - k, :] = x[:, k:, :]
            elif offset > 0:
                shifted[:, offset:, :] = x[:, : seqlen - offset, :]
            else:
                shifted = x
            y = y + shifted * self.mix[idx].to(dtype=x.dtype)[None, None, :]
        return x + y


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def bench_forward(module, x, iters=200, warmup=50):
    module.eval()
    for _ in range(warmup):
        _ = module(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = module(x)
    sync()
    return (time.perf_counter() - t0) / iters


def bench_forward_backward(module, x, iters=100, warmup=20):
    module.train()
    for _ in range(warmup):
        inp = x.clone().detach().requires_grad_(True)
        out = module(inp)
        loss = out.square().mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        inp = x.clone().detach().requires_grad_(True)
        out = module(inp)
        loss = out.square().mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
    sync()
    return (time.perf_counter() - t0) / iters


def main():
    torch.manual_seed(0)
    device = "cuda"

    B, L, D, K = 8, 2048, 768, 4
    x = torch.randn(B, L, D, device=device, dtype=torch.float32, requires_grad=True)

    slow = CanonLayerSlow(D, K).to(device)
    cuda_layer = CanonLayerCUDA(D, K).to(device)
    cuda_layer.mix.data.copy_(slow.mix.data)

    y_ref = slow(x)
    y_cuda = cuda_layer(x)

    print("forward max err:", (y_ref - y_cuda).abs().max().item())

    g = torch.randn_like(y_ref)
    y_ref.backward(g, retain_graph=True)
    grad_x_ref = x.grad.detach().clone()
    grad_mix_ref = slow.mix.grad.detach().clone()

    x.grad = None
    slow.mix.grad = None

    y_cuda.backward(g)
    grad_x_cuda = x.grad.detach().clone()
    grad_mix_cuda = cuda_layer.mix.grad.detach().clone()

    print("grad_x max err :", (grad_x_ref - grad_x_cuda).abs().max().item())
    print("grad_mix max err:", (grad_mix_ref - grad_mix_cuda).abs().max().item())

    x2 = torch.randn(B, L, D, device=device, dtype=torch.float32)
    slow_fwd = bench_forward(slow, x2)
    cuda_fwd = bench_forward(cuda_layer, x2)

    print(f"slow forward   : {slow_fwd * 1000:.3f} ms")
    print(f"cuda forward   : {cuda_fwd * 1000:.3f} ms")
    print(f"speedup        : {slow_fwd / cuda_fwd:.2f}x")

    slow_fb = bench_forward_backward(slow, x2)
    cuda_fb = bench_forward_backward(cuda_layer, x2)

    print(f"slow fwd+bwd   : {slow_fb * 1000:.3f} ms")
    print(f"cuda fwd+bwd   : {cuda_fb * 1000:.3f} ms")
    print(f"speedup        : {slow_fb / cuda_fb:.2f}x")


if __name__ == "__main__":
    main()