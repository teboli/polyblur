#include <torch/extension.h>
#include <math.h>

using namespace torch::indexing;


/* Note: 
 * The recursive filter is actually slow because it cannot be parallized.
 * Thus, this code does not scale for larger images. In general, use the 
 * normalized convolution instead (NC.cpp).
 */


torch::Tensor filter_horizontal(torch::Tensor F,
                                torch::Tensor D,
                                float sigma) {
    // Feedback coefficient (Appendix of our paper)
    float a = exp(-sqrt(2) / sigma);
    auto V = torch::pow(a, D).unsqueeze(0);

    int w = F.sizes()[3];

    int i;
    // Left -> Right filter
    for(i=1; i<w; i++) {
        // F[..., i] += V[..., i] * (F[..., i-1] - F[..., i]);
        F.index_put_( {"...", i}, 
                      F.index({"...", i}) + V.index({"...", i}) * ( F.index({"...", i-1}) - F.index({"...", i}) ) );
    }

    // Right -> Left filter
    for(i=w-2; i>-1; i--) {
        // F[..., i] += V[..., i+1] * (F[..., i+1] - F[..., i]);
        F.index_put_( {"...", i},
                      F.index({"...", i}) + V.index({"...", i+1}) * ( F.index({"...", i+1}) - F.index({"...", i}) ) );
    }

    return F;
}



torch::Tensor recursive_filter(torch::Tensor I,
                               float sigma_s,
                               float sigma_r,
                               int num_iterations) {
    int batch = I.sizes()[0];
    int h = I.sizes()[2];
    int w = I.sizes()[3];

    //  Compute the domain transform
    //  Estimate horizontal and vertical partial derivatives using finite differences
    auto dIcdx = torch::diff(I, 1, /*dim=*/3);
    auto dIcdy = torch::diff(I, 1, /*dim=*/2);

    // Compute the l1-norm distance of neighbor pixels
    auto dIdx = torch::zeros( {batch, h, w}, torch::dtype(torch::kFloat32).device(I.device()) );
    auto dIdy = torch::zeros( {batch, h, w}, torch::dtype(torch::kFloat32).device(I.device()) );

    dIdx.index_put_( /*[:,:,1:]*/{Slice(), Slice(), Slice(1, None)}, 
                     torch::sum(torch::abs(dIcdx), /*dim=*/1) );
    dIdy.index_put_( /*[:,1:,:]*/{Slice(), Slice(1, None), Slice()}, 
                     torch::sum(torch::abs(dIcdy), /*dim=*/1) );


    // Compute the derivatives of the horizontal and vertical domain transforms
    auto dHdx = 1 + sigma_s / sigma_r * dIdx;
    auto dVdy = 1 + sigma_s / sigma_r * dIdy;

    // The vertical pass is performed using a transposed image
    dVdy = dVdy.transpose(1, 2);

    // Perform filtering
    int N = num_iterations;
    auto F = I.clone();

    float sigma_H = sigma_s;
    float sigma_H_i;

    for(int i=0; i < num_iterations; i++) {
        // Compute the sigma value for this iterations (Equation (14) of our paper)
        sigma_H_i = sigma_H * sqrt(3) * pow(2, N - (i + 1)) / sqrt(pow(4, N) - 1);
        
        F = filter_horizontal(F, dHdx, sigma_H_i);
        F = F.transpose(2, 3);

        F = filter_horizontal(F, dVdy, sigma_H_i);
        F = F.transpose(2, 3);
    }

    return F;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("recursive_filter", &recursive_filter, "Domain transform RC");
}

