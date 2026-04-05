#include <torch/extension.h>
#include <vector>

torch::Tensor canon_forward_cuda(torch::Tensor x, torch::Tensor mix);
std::vector<torch::Tensor> canon_backward_cuda(torch::Tensor grad_out, torch::Tensor x, torch::Tensor mix);

torch::Tensor canon_forward(torch::Tensor x, torch::Tensor mix) {
    return canon_forward_cuda(x, mix);
}

std::vector<torch::Tensor> canon_backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor mix) {
    return canon_backward_cuda(grad_out, x, mix);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &canon_forward, "Canon forward (CUDA)");
    m.def("backward", &canon_backward, "Canon backward (CUDA)");
}
