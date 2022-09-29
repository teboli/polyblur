#include <torch/extension.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>



int find_gaussian_quantile(torch::Tensor var2, 
                           torch::Tensor denom, 
                           float threshold) {
    /* The support size is that to obtain threshold with the Gaussian. 
       The greater threshold, the smaller the support */
    return (int) floor(sqrt( var2.index({0}).item<float>() * log(threshold * denom.index({0}).item<float>()) ));
}


torch::Tensor gaussian1d_filter(torch::Tensor std, 
                                         float threshold,
                                         int support_size) {
    // Allocate the kerne: if support_size > 0 or we have a single image in the batch, it overides threshold
    auto denom = sqrt(2*pi) * std;
    auto var2 = -2.0 * std * std;
    int hsk;
    if (support_size == 0 || std.sizes()[0] == 1 || std.sizes()[0] == 3) {
        hsk = find_gaussian_quantile(var2, denom, threshold);
    } else {
        hsk = (support_size-1) / 2 + 1;  // Fixed for batch case
    }
    auto kernel = torch::empty({std.sizes()[0], (hsk-1)*2 + 1}, torch::dtype(torch::kFloat32).device(std.device()) );

    for (int i=0; i<(hsk-1)*2+1; i++) {
        kernel.index_put_({"...", i}, torch::exp((i-hsk+1)*(i-hsk+1) / var2) / denom);
    }

    return kernel;
}






template <typename scalar_t>
__device__ __forceinline__ int64_t clip(scalar_t x, int min_value, int max_value) {
    return (int64_t) fmax(fmin(value, max_value), min_value);
}


template <typename scalar_t>
__global__ void separable_gaussian_xt_convolve2d_kernel() {

}




torch::Tensor separable_gaussian_xt_convolve2d_cuda(torch::Tensor image,
                                                    torch::Tensor sigma,
                                                    torch::Tensor rho,
                                                    torch::Tensor theta,
                                                    float threshold,
                                                    int support_size) {

    // Find the variance along the axes
    auto co = torch::cos(theta);
    auto so = torch::sin(theta);
    auto dot = rho*rho * co*co + sigma*sigma * so*so;
    auto sigma_phi = torch::sqrt(dot);
    auto sigma_x = sigma * rho / sigma_phi;
    auto tan_phi = dot / ( rho*rho - sigma*sigma + 0.00001);
    auto mu = tan_phi;

    // Create th 1D kernels (truncation is achievd wit the treshold parameter
    auto kernel_x   = gaussian1d_filter(sigma_x, threshold, support_size);  // (B,lx)
    auto kernel_phi = gaussian1d_filter(sigma_phi, threshold, support_size);  // (B,lphi)
    const auto support_x = kernel_x.sizes()[1];  // lx
    const auto support_phi_ = kernel_phi.sizes()[1];  // lphi full
    kernel_phi = kernel_phi.index({Slice(), Slice((support_phi_-1) / 2, None)});
    const auto support_phi = kernel_phi.sizes()[1];  // lphi

    // Create the intermediate images
    const auto b = image.sizes()[0];
    const auto h = image.sizes()[1];
    const auto w = image.sizes()[2];

    //Do the 1D convolution along the x-axis
    int i, z;
    int x, y;
    int x_up, x_down;
    namespace F = torch::nn::functional;
    auto img_x = F::pad( image.unsqueeze(0), 
                         F::PadFuncOptions({(support_x-1)/2, (support_x-1)/2, 0, 0}).mode(torch::kReplicate) );
    img_x = F::conv2d( img_x, kernel_x.unsqueeze(1).unsqueeze(1), 
                       F::Conv2dFuncOptions().groups(b) ).squeeze(0);

    //Do the 1D convolution along the t-axis
    auto img_theta = torch::zeros_like(image);  // Output tensor

    auto xm = torch::empty_like(mu, torch::dtype(torch::kFloat32).device(image.device()));
    auto xp = torch::empty_like(mu, torch::dtype(torch::kFloat32).device(image.device()));
    auto xm_f = torch::empty_like(mu, torch::dtype(torch::kInt64).device(image.device()));
    auto xm_c = torch::empty_like(mu, torch::dtype(torch::kInt64).device(image.device()));  // Long arrays
    auto xp_f = torch::empty_like(mu, torch::dtype(torch::kInt64).device(image.device()));
    auto xp_c = torch::empty_like(mu, torch::dtype(torch::kInt64).device(image.device()));
    auto am = torch::empty_like(mu, torch::dtype(torch::kFloat32).device(image.device()));
    auto ap = torch::empty_like(mu, torch::dtype(torch::kFloat32).device(image.device()));
    auto cumsum = torch::empty( {image.sizes()[0]}, 
                                torch::dtype(torch::kFloat32).device(image.device()) );

    //The CUDA code is here
    const int threads = 1024;
    const dim3 blocks(/*TODO*/);

    AT_DISPATCH_FLOATING_TYPES(image.type(), "separable_gaussian_xt_convolve2d_cuda", ([&] {
          /*TODO*/
          }));

    // End, return the filtered image
    return img_theta;
}
