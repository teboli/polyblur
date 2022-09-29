#include <torch/extension.h>

// CUDA forward declarations

torch::Tensor separable_gaussian_xt_convolve2d_cuda(torch::Tensor image,
                                                    torch::Tensor sigma,
                                                    torch::Tensor rho,
                                                    torch::Tensor theta,
                                                    float threshold,
                                                    int support_size);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor separable_gaussian_xt_convolve2d(torch::Tensor image,
                                               torch::Tensor sigma,
                                               torch::Tensor rho,
                                               torch::Tensor theta,
                                               float threshold,
                                               int support_size){
    CHECK_INPUT(image);
    CHECK_INPUT(sigma);
    CHECK_INPUT(rho);
    CHECK_INPUT(theta);

    return separable_gaussian_xt_convolve2d_cuda(image, 
                                                 sigma, 
                                                 rho, 
                                                 theta, 
                                                 threshold, 
                                                 support_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("separable_gaussian_xt_convolve2d", &separable_gaussian_xt_convolve2d, 
        "Convolution with 1D-separable on non-orthogonal Gaussian filters.");
}
