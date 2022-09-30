#include <torch/extension.h>
#include <math.h>

#include <vector>

using namespace torch::indexing;


torch::Tensor find(
    torch::Tensor query, /*(*,W)*/
    int amount=1) {
    // Returns the amount first elements found for a 
    // query along the last axis
    return torch::nonzero(input).index({"...", Slice(None, amount)});
}


torch::Tensor box_filter_horizontal(
    torch::Tensor I,  /*(B,C,H,W)*/
    torch::Tensor xform_domain_position,  /*(B,H,W)*/
    float box_radius) {

    int batch = I.sizes()[0];
    int num_channels = I.sizes()[1];
    int h = I.sizes()[2];
    int w = I.sizes()[3];

    auto options = torch::dtype(torch::kFloat32).device(I.device());

    // Compute the lower and upper limits of the box kernel in the transformed domain
    auto l_pos = xform_domain_position - box_radius;  // (B,H,W)
    auto u_pos = xform_domain_position + box_radius;  // (B,H,W)

    auto l_idx = torch::zeros_like(xform_domain_position);  // (B,H,W)
    auto u_idx = torch::zeros_like(xform_domain_position);  // (B,H,W)

    auto l_pos_row = torch::empty({batch, w}, options);  // (B,W)
    auto u_pos_row = torch::empty({batch, w}, options);  // (B,W)

    auto local_l_idx = torch::empty({batch, w}, options);  // (B,W)
    auto local_u_idx = torch::empty({batch, w}, options);  // (B,W)


    // Find the indices of the pixels associated with the lower and upper limits
    // of the box kernel.
    int row, col, b;
    auto xform_domain_pos_row = torch::empty({batch, w+1}, options);  // (B,W+1)
    xform_domain_pos_row.index_put( {Slice(), w}, torch::inf );  // Last value is always infinity
    for (row=0; row < h; row++){ /*We can remove this for loop*/
        // Update for the given row the tensor
        xform_domain_pos_row.index_put_( {Slice(), Slice(None, w)}, xform_domain_position.index({Slice(), row})); 

        l_pos_row = l_pos.index({Slice(), row});  // (B,W)
        u_pos_row = u_pos.index({Slice(), row});  // (B,W)

        local_l_idx.fill_(0);  // (B,W)
        local_u_idx.fill_(0);  // (B,W)

        local_l_idx.index_put_( {Slice(), 0}, 
            find( xform_domain_pos_row > l_pos_row.index({Slice(), 0}) )
            );  // (B,W)
        local_u_idx.index_put_( {Slice(), 0}, 
            find( xform_domain_pos_row > u_pos_row.index({Slice(), 0}) )
            );  // (B,W)

        for (col=1; col < w; col++) {
            for(b=0; b < batch, b++) {
                local_l_idx.index_put_( {b, col}, 
                    local_l_idx.index({b, col-1}) +
                    find( xform_domain_pos_row.index({b, Slice(local_l_idx.index({b, col-1}), None)}) >
                          l_pos_row.index({b, col}) ) - 1
                    );  // (,) -- scalar to be added for update
                local_u_idx.index_put_( {b, col}, 
                    local_u_idx.index({b,  col-1}) +
                    find( xform_domain_pos_row.index({b, Slice(local_u_idx.index({b, col-1}), None)}) 
                          u_pos_row.index({b, col}) ) - 1
                    );  // (,) -- scalar to be added for update
            }
        }

        l_idx.index_put_({Slice(), row}, local_l_idx);  // (B,H,W)  -- update for the given row
        u_idx.index_put_({Slice(), row}, local_u_idx);  // (B,H,W)  -- update for the given row
    }

    // Compute the box filter using summed area table. 
    auto SAT = torch::zeros({batch, num_channels, h, w+1}, options);  // (B,H,W+1)
    SAT.index_put_({"...", Slice(1, None)}, torch::cumsum(I, 3));  // Cumsum along width dimension
    auto F = torch::zeros_like(I);  // (B,C,H,W)

    for (int c=0, c<num_channels; c++) {
        F.index_put_({Slice(), c}, 
            (SAT.index({"...", u_idx}) - SAT.index({"...", l_idx})) / (u_idx - l_idx) 
            );  // Elementwise division
    }

    return F;
}

torch::Tensor normalized_convolution(
    torch::Tensor img,
    float sigma_s,
    float sigma_r,
    int num_iterations) {

    auto I = img
    auto options = torch::dtype(torch::kFloat32).device(I.device());

    int batch = I.sizes()[0];
    int h = I.sizes()[2];
    int w = I.sizes()[3];

    //  Compute the domain transform
    //  Estimate horizontal and vertical partial derivatives using finite differences
    auto dIcdx = torch::diff(I, 1, /*dim=*/3);
    auto dIcdy = torch::diff(I, 1, /*dim=*/2);

    // Compute the l1-norm distance of neighbor pixels
    auto dIdx = torch::zeros( {batch, h, w}, options );
    auto dIdy = torch::zeros( {batch, h, w}, options );

    dIdx.index_put_( /*[:,:,1:]*/{Slice(), Slice(), Slice(1, None)}, 
                     torch::sum(torch::abs(dIcdx), /*dim=*/1) );
    dIdy.index_put_( /*[:,1:,:]*/{Slice(), Slice(1, None), Slice()}, 
                     torch::sum(torch::abs(dIcdy), /*dim=*/1) );


    // Compute the derivatives of the horizontal and vertical domain transforms
    auto dHdx = 1 + sigma_s / sigma_r * dIdx;
    auto dVdy = 1 + sigma_s / sigma_r * dIdy;

    // Integrate the domain transforms
    auto ct_H = torch::cumsum(dHdx, 3);
    auto ct_V = torch::cumsum(dVdy, 2);

    // The vertical pass is performed using a transposed image
    ct_V = ct_V.transpose(1, 2);

    // Perform filtering
    int N = num_iterations;
    auto F = I.clone();

    float sigma_H = sigma_s;
    float sigma_H_i;
    float box_radius;

    for(int=0; i < num_iterations; i++) {
        // Compute the sigma value for this iteration (Equation (14) of our paper)
        sigma_H_i = sigma_H * sqrt(3) * pow(2, N - (i + 1)) / sqrt(pow(4, N) - 1);

        // Compute the radius of the box filter with thr desired variance
        box_radius = sqrt(3) * sigma_H_i;

        F = box_filter_horizontal(F, ct_H, box_radius);
        F = F.transpose(2, 3);

        F = box_filter_horizontal(F, ct_V, box_radius);
        F = F.transpose(2, 3);
    }

    return F;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("normalized_convolution", &normalized_convolution, "Domain transform NC");
}

