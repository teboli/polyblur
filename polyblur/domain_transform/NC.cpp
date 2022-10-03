#include <torch/extension.h>
#include <math.h>

#include <vector>
#include <iostream>

using namespace torch::indexing;


torch::Tensor find(
    torch::Tensor query /*(B,W) or (W)*/) {
    // Returns the amount first elements found for a 
    // query along the last axis
    if (query.dim() == 1) {
        query = query.unsqueeze(0);
    }

    // First find all the nonzero values
    auto indices = torch::nonzero(query);  // (n,2)

    // Now find for each element of the batch the first nonzero index
    int batch = query.sizes()[0];
    auto compact_indices = torch::zeros(batch, torch::dtype(torch::kInt32).device(query.device()));
    // If no zero detected in torch::nonzero, simply return zero
    if (0 == indices.sizes()[0]) {
        return compact_indices;
    // Else, do the search
    } else {
        int iter;
        bool first_batch_idx_detected;
        for (int b = 0; b < batch; b++) {
            iter = 0;
            first_batch_idx_detected = false;
            // while we have not reach the end of the array or the batch index is not detected yet
            while ((iter < batch) || (!first_batch_idx_detected)) {
                // If we reach the first index that matches the current batch value, 
                // store the correspoding width
                if (b == indices.index({iter, 0}).item<int>()) {
                    compact_indices.index_put_({b}, indices.index({iter, 1}));
                    first_batch_idx_detected = true;  // Put an end to the loop
                }
                iter++;
            }  // Now reset iter and first_batch_idx_detected
        }
        return compact_indices;  // (B) or (1)
    }
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
    auto options_int = torch::dtype(torch::kInt64).device(I.device());

    // Compute the lower and upper limits of the box kernel in the transformed domain
    auto l_pos = xform_domain_position - box_radius;  // (B,H,W)
    auto u_pos = xform_domain_position + box_radius;  // (B,H,W)

    auto l_idx = torch::ones({batch,h,w}, options_int);  // (B,H,W)
    auto u_idx = 2* torch::ones({batch,h,w}, options_int);  // (B,H,W)

    auto l_pos_row = torch::zeros({batch, w}, options_int);  // (B,W) -- shared
    auto u_pos_row = torch::zeros({batch, w}, options_int);  // (B,W) -- shared

    auto local_l_idx = torch::zeros({batch, w}, options_int);  // (B,W) -- shared
    auto local_u_idx = torch::zeros({batch, w}, options_int);  // (B,W) -- shared


    // Find the indices of the pixels associated with the lower and upper limits
    // of the box kernel.
    int row, col, b;

    auto xform_domain_pos_row = torch::zeros({batch, w+1}, options);  // (B,W+1) -- sharred
    xform_domain_pos_row.index_put_( {Slice(), w}, pow(2, 16)-1);  // Last value is always infinity

    // Note: Must be compiled with multithreading library for faster for loop
    #pragma omp parallel for  
    for (row=0; row < h; row++){
        // Update for the given row the tensor
        // xform_domain_pos_row[:,:w] = xform_domain_position[:, row, :]
        // should be splitted
        xform_domain_pos_row.index_put_( {Slice(), Slice(None, w)}, xform_domain_position.index({Slice(), row}));  // put (B,W) in (B,W+1)

        l_pos_row = l_pos.index({Slice(), row, Slice()});  // (B,W) -- should be splitted
        u_pos_row = u_pos.index({Slice(), row, Slice()});  // (B,W) -- should be splitted

        local_l_idx.fill_(0);  // (B,W) -- should be splitted
        local_u_idx.fill_(0);  // (B,W) -- should be splitted

        local_l_idx.index_put_( {Slice(), 0}, find( xform_domain_pos_row > l_pos_row.index({Slice(), 0}) ) );  // (B,W)
        local_u_idx.index_put_( {Slice(), 0}, find( xform_domain_pos_row > u_pos_row.index({Slice(), 0}) ) );  // (B,W)

        for (col=1; col < w; col++) {
            for (b=0; b < batch; b++) {
                // local_l_idx[b, col] = local_l_idx[b, col-1] + find(xform_domain_pos_row[b, local_l_idx[b, col-1]:] > l_pos_row[b, col]) - 1
                local_l_idx.index_put_( {b, col}, local_l_idx.index({b, col-1}) +
                    find( xform_domain_pos_row.index({b, Slice( local_l_idx.index({b, col-1}).item<int>(), None )}) > l_pos_row.index({b, col}) ).item<int>() );  // (,) -- scalar to be added for update
                // local_u_idx[b, col] = local_u_idx[b, col-1] + find(xform_domain_pos_row[b, local_u_idx[b, col-1]:] > u_pos_row[b, col]) - 1
                local_u_idx.index_put_( {b, col}, local_u_idx.index({b, col-1}) +
                    find( xform_domain_pos_row.index({b, Slice( local_u_idx.index({b, col-1}).item<int>(), None )}) > u_pos_row.index({b, col}) ).item<int>() );  // (,) -- scalar to be added for update
            }
        }
  
        l_idx.index_put_({Slice(), row}, local_l_idx);  // update (B,row,W)  -- update for the given row
        u_idx.index_put_({Slice(), row}, local_u_idx);  // update (B,row,W)  -- update for the given row
    }
    // End of parallel for loop

    // Compute the box filter using summed area table. 
    auto SAT = torch::zeros({batch, num_channels, h, w+1}, options);  // (B,C,H,W+1)
    SAT.index_put_({"...", Slice(1, None)}, torch::cumsum(I, 3));  // Cumsum along width dimension
    auto F = torch::zeros_like(I).reshape({batch, num_channels, h*w});  // (B,C,H*W)

    auto u_idx_linear = u_idx + (w+1) * torch::arange(h, options_int).view({h,1});  //  (B,H,W)
    u_idx_linear = u_idx_linear.reshape({batch, h*w});  //  (B,H*W)
    auto l_idx_linear = l_idx + (w+1) * torch::arange(h, options_int).view({h,1});  //  (B,H,W)
    l_idx_linear = l_idx_linear.reshape({batch, h*w});  //  (B,H*W)
    SAT = SAT.view({batch, num_channels, h*(w+1)});  // Linear spatial indexing search -- (B,C,H*(W+1))
    auto b_idx_linear = torch::arange( batch, options_int );  // (B,1)
    // auto b_idx_linear = torch::arange( batch, options_int ).view({batch,1});  // (B,1)

    // Elementwise division of (B,H*W) arrays + view to 2D coordinate grid (B,H,W) -- the correctness of slicing was checked in python
    F.index_put_( {Slice(), 0}, SAT.index({b_idx_linear, 0, u_idx_linear}) - SAT.index({b_idx_linear, 0, l_idx_linear}) );  // (B,H,W) slices
    F.index_put_( {Slice(), 1}, SAT.index({b_idx_linear, 1, u_idx_linear}) - SAT.index({b_idx_linear, 1, l_idx_linear}) );  // (B,H,W) slices      
    F.index_put_( {Slice(), 2}, SAT.index({b_idx_linear, 2, u_idx_linear}) - SAT.index({b_idx_linear, 2, l_idx_linear}) );  // (B,H,W) slices      

    // Go back to (B,C,H,W) image format and return
    F = F.reshape({batch, num_channels, h, w});
    F = F / (u_idx - l_idx + 0.0001);

    return F;
}


torch::Tensor normalized_convolution(
    torch::Tensor I,
    float sigma_s,
    float sigma_r,
    int num_iterations) {

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
    auto ct_H = torch::cumsum(dHdx, /*dim=*/2);
    auto ct_V = torch::cumsum(dVdy, /*dim=*/1);

    // The vertical pass is performed using a transposed image
    ct_V = ct_V.transpose(1, 2);

    // Perform filtering
    int N = num_iterations;
    auto F = I.clone();

    float sigma_H = sigma_s;
    float sigma_H_i;
    float box_radius;

    for(int i=0; i < num_iterations; i++) {
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

