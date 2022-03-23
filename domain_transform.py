import numpy as np
import torch


def recursive_filter(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    if img.dtype == np.ndarray:
        return recursive_filter_np(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None)
    else:
        return recursive_filter_torch(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None)


def recursive_filter_np(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    if img.ndim ==2:
        img = img[..., None]
    I = np.array(img)

    if joint_image is None:
        J = I
    else:
        J = joint_image

    h, w, num_joint_channels = J.shape

    ## Compute the domain transform
    # Estimate horizontal and vertical partial derivatives using finite differences
    dIcdx = np.diff(J, n=1, axis=1)
    dIcdy = np.diff(J, n=1, axis=0)

    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    # compute the l1-norm distance of neighbor pixels.
    for c in range(num_joint_channels):
        dIdx[:, 1:] = dIdx[:, 1:] + np.abs(dIcdx[:, :, c])
        dIdy[1:, :] = dIdy[1:, :] + np.abs(dIcdy[:, :, c])

    # compute the derivatives of the horizontal and vertical domain transforms
    dHdx = (1 + sigma_s/sigma_r * dIdx)
    dVdy = (1 + sigma_s/sigma_r * dIdy)

    # the veritcal pass is performed using a transposed image
    dVdy = dVdy.T

    ## Perform the filtering
    N = num_iterations
    F = I

    sigma_H = sigma_s

    for i in range(num_iterations):
        # Compute the sigma value for this iterations (Equation 14 of our paper)
        sigma_H_i = sigma_H * np.sqrt(3) * 2**(N - (i + 1)) / np.sqrt(4**N - 1)

        F = transformed_domain_recursive_filter_horizontal_np(F, dHdx, sigma_H_i)
        F = np.transpose(F, (1, 0, 2))

        F = transformed_domain_recursive_filter_horizontal_np(F, dVdy, sigma_H_i)
        F = np.transpose(F, (1, 0, 2))

    F = F.astype(img.dtype)

    return np.squeeze(F, -1)  # for grayscale images


def transformed_domain_recursive_filter_horizontal_np(I, D, sigma):
    # Feedback coefficient (Appendix of our paper).
    a = np.exp(-np.sqrt(2) / sigma)

    F = I
    V = a**D

    h, w, num_channels = I.shape

    # Left -> Right filter
    for i in range(1, w, 1):
        for c in range(num_channels):
            F[:, i, c] = F[:, i, c] + V[:, i] * (F[:, i - 1, c] - F[:, i, c])

    # Right -> Left filter
    for i in range(w-2, -1, -1):  # from w-2 to 0
        for c in range(num_channels):
            F[:, i, c] = F[:, i, c] + V[:, i + 1] * (F[:, i + 1, c] - F[:, i, c])

    return F


def recursive_filter_torch(img, sigma_s=60, sigma_r=0.4, num_iterations=3, joint_image=None):
    I = img.clone()

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
    dIdy = torch.sum(torch.abs(dIcdy), dim=1)

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
        sigma_H_i = sigma_H * np.sqrt(3) * 2**(N - (i + 1)) / np.sqrt(4**N - 1)

        F = transformed_domain_recursive_filter_horizontal_torch(F, dHdx, sigma_H_i)
        F = F.transpose(-1, -2)

        F = transformed_domain_recursive_filter_horizontal_torch(F, dVdy, sigma_H_i)
        F = F.transpose(-1, -2)

    F = F.type_as(img)

    return F


def transformed_domain_recursive_filter_horizontal_torch(I, D, sigma):
    # Feedback coefficient (Appendix of our paper).
    a = np.exp(-np.sqrt(2) / sigma)

    F = I
    V = a**D

    batch, num_channels, h, w = I.shape

    # Left -> Right filter
    for i in range(1, w, 1):
        for c in range(num_channels):
            F[:, c, :, i] = F[:, c, :, i] + V[:, :, i] * (F[:, c, :, i - 1] - F[:, c, :, i])

    # Right -> Left filter
    for i in range(w-2, -1, -1):  # from w-2 to 0
        for c in range(num_channels):
            F[:, c, :, i] = F[:, c, :, i] + V[:, :, i + 1] * (F[:, c, :, i + 1] - F[:, c, :, i])

    return F


if __name__ == '__main__':
    from skimage import data, img_as_float32
    import matplotlib.pyplot as plt
    import cv2

    img = img_as_float32(data.astronaut())
    noise = 0.04 * np.random.randn(*img.shape)
    img_noise = np.clip(img + noise, 0, 1)

    sigma_r = 0.4
    sigma_s = 60

    img_filtered = recursive_filter(img_noise, num_iterations=1, sigma_s=sigma_s, sigma_r=sigma_r)

    img_cv2 = cv2.edgePreservingFilter((img_noise * 255).astype(np.uint8), flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    img_cv2 = img_as_float32(img_cv2)

    diff = img_noise - img_filtered

    plt.figure(figsize=(20, 6))
    plt.subplot(1,3,1)
    plt.imshow(img_noise)
    plt.subplot(1,3,2)
    plt.imshow(img_filtered)
    plt.subplot(1, 3, 3)
    plt.imshow(img_cv2)
    plt.show()

    print(np.linalg.norm(img_cv2 - img_filtered))
