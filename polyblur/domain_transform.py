import torch
import math


def recursive_filter(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    """
    (pytorch) Implementation of the edge aware smoothing with recursive filtering (EdgeAwareSmoothing Alg.6) from:
        [Eduardo Simoes Lopes Gastal and Manuel M. Oliveira. Domain transform for edge-aware
         image and video processing. ACM Transactions on Graphics (ToG), 30(4):69, 2011.]
    :param img: (B,C,H,W) torch.tensor, the input image(s)
    :param sigma_r: float, regularization parameter for domain transform 
    :param sigma_s: float, smoothness parameter for domain transform 
    :param num_iterations: int, iterations
    :param joint_image: (B,C,H,W) torch.tensor, the guide image(s) (optional)
    :return: img_smoothed: torch.tensor of same size as img, the smoothed image(s)
    """
    I = img

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
    dIdx = torch.zeros(batch, h, w, device=I.device)
    dIdx[:, :, 1:] = torch.sum(torch.abs(dIcdx), dim=1)
    dIdy = torch.zeros(batch, h, w, device=I.device)
    dIdy[:, 1:, :] = torch.sum(torch.abs(dIcdy), dim=1)

    # compute the derivatives of the horizontal and vertical domain transforms
    dHdx = (1 + sigma_s/sigma_r * dIdx)
    dVdy = (1 + sigma_s/sigma_r * dIdy)

    # the vertical pass is performed using a transposed image
    dVdy = dVdy.transpose(-2, -1)

    ## Perform the filtering
    N = num_iterations
    F = I

    sigma_H = sigma_s

    for i in range(num_iterations):
        # Compute the sigma value for this iterations (Equation 14 of our paper)
        sigma_H_i = sigma_H * math.sqrt(3) * 2**(N - (i + 1)) / math.sqrt(4**N - 1)

        F = transformed_domain_recursive_filter_horizontal(F, dHdx, sigma_H_i)
        F = F.transpose(-1, -2)

        F = transformed_domain_recursive_filter_horizontal(F, dVdy, sigma_H_i)
        F = F.transpose(-1, -2)

    return F


def transformed_domain_recursive_filter_horizontal(I, D, sigma):
    """
    (pytorch) Implementation of the recursive 1D (horizontal) filtering (Recursive1DFilter Alg.7) used in the edge aware smoothing from:
        [Eduardo Simoes Lopes Gastal and Manuel M. Oliveira. Domain transform for edge-aware
         image and video processing. ACM Transactions on Graphics (ToG), 30(4):69, 2011.]
    :param I: (B,C,H,W) torch.tensor, the input image(s)
    :param D: (B,C,H,W) torch.tensor, the image of "distances(s)" used to control the diffusion
    :param sigma: float, regularization parameter 
    :return: img_smoothed: torch.tensor of same size as img, the filtered image(s)
    """
    # Feedback coefficient (Appendix of our paper).
    a = math.exp(-math.sqrt(2) / sigma)

    F = I 

    V = a**D
    V = V.unsqueeze(1)

    batch, num_channels, h, w = F.shape

    # Left -> Right filter
    for i in range(1, w, 1):
        F[..., i] += V[..., i] * (F[..., i - 1] - F[..., i])

    # Right -> Left filter
    for i in range(w-2, -1, -1):  # from w-2 to 0
        F[..., i] += V[..., i + 1] * (F[..., i + 1] - F[..., i])

    return F

