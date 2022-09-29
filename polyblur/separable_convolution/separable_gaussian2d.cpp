#include <torch/extension.h>
#include <math.h>
#include <iostream>

using namespace torch::indexing;


int64_t clip(int value, int min_value, int max_value) {
    return (int64_t) fmax(fmin(value, max_value), min_value);
}


const float pi = M_PI;

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



torch::Tensor separable_gaussian_ortho_convolve2d(torch::Tensor image,
                                                  torch::Tensor sigma,
                                                  torch::Tensor rho,
                                                  torch::Tensor theta,
                                                  float threshold, 
                                                  int support_size){
    // Setup the 1D stds
    auto sigma_y = sigma.clone();
    auto sigma_x = rho.clone();
 
    auto mask = torch::fmod(torch::floor(theta * 180 / pi), 180) < 0.0001;
    if ( torch::any(mask).item<bool>() ) {
        sigma_y.index_put_({mask}, rho.index({mask}));
        sigma_x.index_put_({mask}, sigma.index({mask}));
    }

    // Create th 1D kernels (truncation is achievd wit the treshold parameter
    auto kernel_x = gaussian1d_filter(sigma_x, threshold, support_size);  // (B,lx)
    auto kernel_y = gaussian1d_filter(sigma_y, threshold, support_size);  // (B,ly)
    int support_x = kernel_x.sizes()[1];  // lx
    int support_y = kernel_y.sizes()[1];  // ly

    // Create the intermediate images
    int b = image.sizes()[0];
    int h = image.sizes()[1];
    int w = image.sizes()[2];
 
    //Do the 1D convolution along the x-axis
    namespace F = torch::nn::functional;
    auto img_x = F::pad( image.unsqueeze(0), 
                         F::PadFuncOptions({(support_x-1)/2, (support_x-1)/2, 0, 0}).mode(torch::kReplicate) );
    img_x = F::conv2d( img_x, kernel_x.unsqueeze(1).unsqueeze(1), 
                       F::Conv2dFuncOptions().groups(b) ).squeeze(0);

    //Do the 1D convolution along the x-axis
    auto img_y = F::pad( img_x.unsqueeze(0), 
                         F::PadFuncOptions({0, 0, (support_y-1)/2, (support_y-1)/2}).mode(torch::kReplicate) );
    img_y = F::conv2d( img_y, kernel_y.unsqueeze(1).unsqueeze(3), 
                       F::Conv2dFuncOptions().groups(b) ).squeeze(0);

    return img_y;
}


torch::Tensor separable_gaussian_xt_convolve2d(torch::Tensor image,
                                               torch::Tensor sigma,
                                               torch::Tensor rho,
                                               torch::Tensor theta,
                                               float threshold,
                                               int support_size){
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
    int support_x = kernel_x.sizes()[1];  // lx
    int support_phi = kernel_phi.sizes()[1];  // lphi
    kernel_phi = kernel_phi.index({Slice(), Slice((support_phi-1) / 2, None)});
    support_phi = kernel_phi.sizes()[1];

    // Create the intermediate images
    int b = image.sizes()[0];
    int h = image.sizes()[1];
    int w = image.sizes()[2];

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
    auto img_theta = torch::zeros_like(image);

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

    // auto t = torch::arange(batch, torch::dtype(torch::kInt64).device(image.device()));
    int t;
    for (z = 0; z < h * w; z++) {
        // From the linear to 2D indices
        x = fmod(z, w);
        y = (z - x) / w;
  
        // Central position in the support
        img_theta.index_put_({"...", y, x}, 
                             kernel_phi.index({"...", 0}) * img_x.index({"...", y, x}));

        // Iteration on the kernel support (with 'edge' padding)
        for (i = 1; i < support_phi; i++) {
          // Get the indices involved
          xm = x - i / mu;  // the left value (m is for 'minus')
          xm_f = torch::floor(xm).to(torch::kInt64);
          xm_c = torch::ceil(xm).to(torch::kInt64);
          xp = x + i / mu;  // the right value (p is for 'plus')
          xp_f = torch::floor(xp).to(torch::kInt64);
          xp_c = torch::ceil(xp).to(torch::kInt64);

          // Bilinear interpolation coefficients
          am = (xm_c - xm) / (xm_c - xm_f);
          ap = (xp_c - xp) / (xp_c - xp_f);

          // Bilinear interpolation
          for(t=0; t < b; t++){
              cumsum[t] =  am[t]     * img_x.index({t, clip(y-i,0,h-1), clip(xm_f[t],0,w-1)});
              cumsum[t] += (1-am[t]) * img_x.index({t, clip(y-i,0,h-1), clip(xm_c[t],0,w-1)});
              cumsum[t] += ap[t]     * img_x.index({t, clip(y+i,0,h-1), clip(xp_f[t],0,w-1)});
              cumsum[t] += (1-ap)[t] * img_x.index({t, clip(y+i,0,h-1), clip(xp_c[t],0,w-1)});
          }

          // Accumulation
          img_theta.index_put_({"...", y, x},
                             img_theta.index({"...", y, x}) + kernel_phi.index({"...", i}) * cumsum );
        }
    }

    return img_theta;
}


torch::Tensor separable_gaussian_convolve2d(torch::Tensor image,
                                            torch::Tensor sigma,
                                            torch::Tensor rho,
                                            torch::Tensor theta,
                                            float threshold,
                                            int support_size){
    /* Separable convolution for 2D Gaussian
       Inputs:
           image: (B,C,H,W), the images
           sigma: (B,C) or (B, 1), the stds in the main direction
           rho: (B, C) or (B, 1), the stds in the tangential direction
           theta: (B, C) or (B, 1) the orientation in the main direction
       Output:
           imout: (B,C,H,W), the filtered images
    */
    int batch = image.sizes()[0];
    int channel = image.sizes()[1];
    int h = image.sizes()[2];
    int w = image.sizes()[3];

    // If kernels are (B,1), make 'em (B,C)
    if (sigma.sizes()[1] != channel) {
        sigma = sigma.repeat({1, channel});
        rho = rho.repeat({1, channel});
        theta = theta.repeat({1, channel});
    }

    // Reshape the images as (B*C,H,W) and the kernels as (B*C)
    image = image.view({batch*channel, h, w});
    sigma = sigma.view({batch*channel});
    rho = rho.view({batch*channel});
    theta = theta.view({batch*channel});
    auto imout = image.clone();
  
    // First check the orthogonal kernels - same variances or theta % 90
    float atol = 0.0001;
    auto mask_ortho = torch::bitwise_or(torch::fmod(theta * 180 / pi, 90) <= atol, 
                                        torch::eq(sigma, rho));
    if ( torch::any(mask_ortho).item<bool>() ) {
        imout.index_put_({mask_ortho}, 
                          separable_gaussian_ortho_convolve2d( image.index({mask_ortho}),
                                                               sigma.index({mask_ortho}),
                                                               rho.index({mask_ortho}),
                                                               theta.index({mask_ortho}),
                                                               threshold,
                                                               support_size ));
    }

    // Otherwise, we do the xt transform
    auto mask_xt = torch::bitwise_and(torch::fmod(theta * 180 / pi, 90) > atol, 
                                      torch::ne(sigma, rho));
    if ( torch::any(mask_xt).item<bool>() ) {
        imout.index_put_({mask_xt},
                      separable_gaussian_xt_convolve2d( image.index({mask_xt}),
                                                        sigma.index({mask_xt}),
                                                        rho.index({mask_xt}),
                                                        theta.index({mask_xt}),
                                                        threshold,
                                                        support_size ));
    }

    return imout.view({batch,channel,h,w});
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("separable_gaussian_convolve2d", &separable_gaussian_convolve2d, 
      "Convolution with 1D-separable, possibly non-orthogonal Gaussian filters.");
}

