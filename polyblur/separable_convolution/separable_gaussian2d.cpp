#include <torch/extension.h>
#include <math.h>

using namespace torch::indexing;


int clip(int value, int min_value, int max_value) {
    return fmax(fmin(value, max_value), min_value);
}


const float pi = M_PI;

int find_gaussian_quantile(torch::Tensor var2, 
                                     torch::Tensor denom, 
                                     float threshold) {
    // return torch::floor(torch::sqrt(var2 * torch::log(threshold * denom)));
    return 1;
}


torch::Tensor gaussian_filter_convolve1d(torch::Tensor std, 
                                         float threshold,
                                         int support_size) {
    // Allocate the kerne: if support_size > 0, it overide threshold
    auto denom = sqrt(2*pi) * std;
    auto var2 = -2.0 * std * std;
    int hsk;
    if (support_size == 0) {
        hsk = find_gaussian_quantile(var2, denom, threshold);
    } else {
        hsk = support_size;  // Fixed for batch case
    }
    auto kernel = torch::empty({std.sizes()[0], std.sizes()[1], hsk});

    // Build the kernel
    for (int i=0; i<hsk; i++) {
        kernel.index_put_({"...", i}, torch::exp(i*i / var2)) / denom;
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
    auto sigma_y = sigma;
    auto sigma_x = rho;
    
    auto mask = torch::eq( torch::fmod(torch::floor(theta * 180 / pi), 180), 0 );
    sigma_y.index_put_({mask}, rho.index({mask}));
    sigma_x.index_put_({mask}, sigma.index({mask}));
  
    // Create th 1D kernels (truncation is achievd wit the treshold parameter
    auto kernel_x = gaussian_filter_convolve1d(sigma_x, threshold, support_size);  // (B,C,lx)
    auto kernel_y = gaussian_filter_convolve1d(sigma_y, threshold, support_size);  // (B,C,ly)
    int support_x = kernel_x.sizes()[2];  // lx
    int support_y = kernel_y.sizes()[2];  // ly
  
    // Create the intermediate images
    int h = image.sizes()[2];
    int w = image.sizes()[3];
    auto img_x = torch::zeros_like(image);
    auto img_y = torch::zeros_like(image);
  
    //Do the 1D convolution along the x-axis
    int i, z;
    int x, y;
    int x_up, x_down;
    for (z = 0; z < h * w; z++) {
        // From the linear to 2D indices
        x = fmod(z, w);
        y = (z - x) / w;
  
        // Central position in the support
        img_x.index_put_({"...", y, x}, 
                         kernel_x.index({"...", 0}) * image.index({"...", y, x}));
  
        // Iteration on the kernel support (with 'edge' padding)
        for (i = 1; i < support_x; i++) {
            x_down = fmax(x-i, 0);
            x_up = fmin(x+i, w-1);
            // Use the fact that the kernel is symmetric to do just one multiplication
            img_x.index_put_({"...", y, x},
                             kernel_x.index({"...", 0}) + 
                             kernel_x.index({"...", i}) * ( image.index({"...", y, x_down}) - 
                                                            image.index({"...", y, x_up}) ) );
        }
    }

    //Do the 1D convolution along the x-axis
    int y_up, y_down;
    for (z = 0; z < h * w; z++) {
        // From the linear to 2D indices
        x = fmod(z, w);
        y = (z - x) / w;
  
        // Central position in the support
        img_y.index_put_({"...", y, x}, 
                         kernel_y.index({"...", 0}) * img_x.index({"...", y, x}));
  
        // Iteration on the kernel support (with 'edge' padding)
        for (i = 1; i < support_y; i++) {
            y_down = fmax(y-i,0);
            y_up = fmin(y+1,h-1);
            // Use the fact that the kernel is symmetric to do just one multiplication
            img_y.index_put_({"...", y, x},
                             kernel_y.index({"...", 0}) + 
                             kernel_y.index({"...", i}) * ( img_x.index({"...", y_down, x}) - 
                                                            img_x.index({"...", y_up, x}) ) );
        }
    }
 
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
    auto kernel_x = gaussian_filter_convolve1d(sigma_x, threshold, support_size);  // (B,C,lx)
    auto kernel_phi = gaussian_filter_convolve1d(sigma_phi, threshold, support_size);  // (B,C,lphi)
    int support_x = kernel_x.sizes()[2];  // lx
    int support_phi = kernel_phi.sizes()[2];  // lphi
  
    // Create the intermediate images
    int h = image.sizes()[2];
    int w = image.sizes()[3];
    auto img_x = torch::zeros_like(image);
    auto img_theta = torch::zeros_like(image);

    //Do the 1D convolution along the x-axis
    int i, z;
    int x, y;
    int x_up, x_down;
    for (z = 0; z < h * w; z++) {
        // From the linear to 2D indices
        x = fmod(z, w);
        y = (z - x) / w;
  
        // Central position in the support
        img_x.index_put_({"...", y, x}, 
                         kernel_x.index({"...", 0}) * image.index({"...", y, x}));
  
        // Iteration on the kernel support (with 'edge' padding)
        for (i = 1; i < support_x; i++) {
            x_down = fmax(x-i, 0);
            x_up = fmin(x+i, w-1);
            // Use the fact that the kernel is symmetric to do just one multiplication
            img_x.index_put_({"...", y, x},
                             kernel_x.index({"...", 0}) + 
                             kernel_x.index({"...", i}) * ( image.index({"...", y, x_down}) - 
                                                            image.index({"...", y, x_up}) ) );
        }
    }
    
    //Do the 1D convolution along the t-axis
    auto xm = torch::empty_like(mu);
    auto xm_f = torch::empty_like(mu);
    auto xm_c = torch::empty_like(mu);
    auto xp = torch::empty_like(mu);
    auto xp_f = torch::empty_like(mu);
    auto xp_c = torch::empty_like(mu);
    auto am = torch::empty_like(mu);
    auto ap = torch::empty_like(mu);
    auto cumsum = torch::empty( {image.sizes()[0], image.sizes()[1]}, 
                                torch::dtype(torch::kFloat32).device(image.device()) );

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
          xm_f = torch::floor(xm);
          xm_c = torch::ceil(xm);
          xp = x + i / mu;  // the right value (p is for 'plus')
          xp_f = torch::floor(xp);
          xp_c = torch::ceil(xp);

          // Bilinear interpolation coefficients
          am = (xm_c - xm) / (xm_c - xm_f);
          ap = (xp_c - xp) / (xp_c - xp_f);

          // Bilinear interpolation
          cumsum =  am     * img_x.index({"...", clip(y-i,0,h-1), clip(xm_f,0,w-1)});
          cumsum += (1-am) * img_x.index({"...", clip(y-i,0,h-1), clip(xm_c,0,w-1)});
          cumsum += ap     * img_x.index({"...", clip(y+i,0,h-1), clip(xp_f,0,w-1)});
          cumsum += (1-ap) * img_x.index({"...", clip(y+i,0,h-1), clip(xp_c,0,w-1)});

          // Accumulation
          img_theta.index_put_({"...", y, x},
                             kernel_phi.index({"...", 0}) + kernel_phi.index({"...", i}) * cumsum );
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
    auto imout = image.clone();
  
    // First check the orthogonal kernels - same variances or theta % 90
    auto mask_ortho = torch::bitwise_or(torch::eq(torch::fmod(theta * 180 / pi, 90), 0), 
                                        torch::eq(sigma, rho));
    imout.index_put_({mask_ortho}, 
                      separable_gaussian_ortho_convolve2d( image.index({mask_ortho}),
                                                           sigma.index({mask_ortho}),
                                                           rho.index({mask_ortho}),
                                                           theta.index({mask_ortho}),
                                                           threshold,
                                                           support_size ));
    // Otherwise, we do the xt transform
    auto mask_xt = torch::bitwise_or(torch::ne(torch::fmod(theta * 180 / pi, 90), 0), 
                                        torch::ne(sigma, rho));
    imout.index_put_({mask_xt},
                      separable_gaussian_xt_convolve2d( image.index({mask_xt}),
                                                        sigma.index({mask_xt}),
                                                        rho.index({mask_xt}),
                                                        theta.index({mask_xt}),
                                                        threshold,
                                                        support_size ));
    return imout;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("separable_gaussian_convolve2d", &separable_gaussian_convolve2d, 
      "Convolution with 1D-separable, possibly non-orthogonal Gaussian filters.");
}

