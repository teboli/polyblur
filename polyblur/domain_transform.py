import math

import torch


def recursive_filter(I, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    """
    (pytorch) Implementation of the edge aware smoothing with recursive filtering (EdgeAwareSmoothing Alg.6) from:
        [Eduardo Simoes Lopes Gastal and Manuel M. Oliveira. Domain transform for edge-aware
         image and video processing. ACM Transactions on Graphics (ToG), 30(4):69, 2011.]
    :param I: (B,C,H,W) torch.tensor, the input image(s)
    :param sigma_r: float, regularization parameter for domain transform 
    :param sigma_s: float, smoothness parameter for domain transform 
    :param num_iterations: int, iterations
    :param joint_image: (B,C,H,W) torch.tensor, the guide image(s) (optional)
    :return: img_smoothed: torch.tensor of same size as img, the smoothed image(s)
    """
    if joint_image is None:
        J = I
    else:
        J = joint_image

    batch, num_joint_channels, h, w = J.shape

    ## Compute the domain transform
    # Estimate horizontal and vertical partial derivatives using finite differences
    dIcdx = torch.diff(J, n=1, dim=-1)
    dIcdy = torch.diff(J, n=1, dim=-2)

    # compute the l1-norm distance of neighbor pixels.
    dIdx = torch.sum(torch.abs(dIcdx), dim=1)
    dIdx = torch.nn.functional.pad(dIdx.unsqueeze(0), pad=(1,0,0,0)).squeeze(0)
    dIdy = torch.sum(torch.abs(dIcdy), dim=1)
    dIdy = torch.nn.functional.pad(dIdy.unsqueeze(0), pad=(0,0,1,0)).squeeze(0)

    # compute the derivatives of the horizontal and vertical domain transforms
    dHdx = 1 + sigma_s/sigma_r * dIdx
    dVdy = 1 + sigma_s/sigma_r * dIdy

    # the vertical pass is performed using a transposed image
    dVdy = dVdy.transpose(-2, -1)

    ## Perform the filtering
    N = num_iterations
    F = I.clone()

    sigma_H = sigma_s
    for i in range(num_iterations):
        # Compute the sigma value for this iterations (Equation 14 of our paper)
        sigma_H_i = sigma_H * math.sqrt(3) * 2**(N - (i + 1)) / math.sqrt(4**N - 1)

        # Feedback coefficient (Appendix of our paper).
        a = math.exp(-math.sqrt(2) / sigma_H_i)
        
        V = (a**dHdx).unsqueeze(1)
        F = transformed_domain_recursive_filter_horizontal(F, V)
        F = F.transpose(-1, -2)

        V = (a**dVdy).unsqueeze(1)
        F = transformed_domain_recursive_filter_horizontal(F, V)
        F = F.transpose(-1, -2)

    return F


@torch.jit.script
def transformed_domain_recursive_filter_horizontal(F, V):
    """
    (pytorch) Implementation of the recursive 1D (horizontal) filtering (Recursive1DFilter Alg.7) used in the edge aware smoothing from:
        [Eduardo Simoes Lopes Gastal and Manuel M. Oliveira. Domain transform for edge-aware
         image and video processing. ACM Transactions on Graphics (ToG), 30(4):69, 2011.]
    :param F: (B,C,H,W) torch.tensor, the input image(s)
    :param D: (B,1,H,W) torch.tensor, the filter used to control the diffusion
    :return: img_smoothed: torch.tensor of same size as img, the filtered image(s)
    """

    # Left -> Right filter
    for i in range(1, F.shape[-1], 1):
        F[..., i] += V[..., i] * (F[..., i - 1] - F[..., i])

    # Right -> Left filter
    for i in range(F.shape[-1]-2, -1, -1):  # from w-2 to 0
        F[..., i] += V[..., i + 1] * (F[..., i + 1] - F[..., i])

    return F

