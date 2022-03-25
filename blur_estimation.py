import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import color
from scipy import interpolate

import utils
from filters import gaussian_filter, fourier_gradients


##############################################
################# Numpy ######################
##############################################


def gaussian_blur_estimation(img, q=0.0001, n_angles=6, n_interpolated_angles=30, c=89.8, b=0.764, ker_size=25,
                             handling_saturation=False):
    """
    Compute an approximate Gaussian filter from a RGB or grayscale image.
    :param img: (H,W,3) or (H,W) array, the image
    :param q: the quantile value used for image normalization
    :param n_angles: the number of angles to compute the directional gradients
    :param n_interpolated_angles: the number of angles to interpolate the directional gradients
    :param c: the slope of the affine model
    :param b: the intercept of the affine model
    :param ker_size: the size of the kernel support
    :return: kernel: the (ker_size,ker_size) approximate Gaussian kernel
    """
    # if img is color, go to grayscale
    if img.ndim == 3:
        img = color.rgb2gray(img)
    # normalized image
    img_normalized = normalize(img, q=q)
    # compute the image gradients
    gradients = compute_gradients(img_normalized)
    # compute the gradient magnitudes per orientation
    gradient_magnitudes = compute_gradient_magnitudes(gradients, n_angles=n_angles)
    # find the maximal blur direction amongst sampled orientations
    magnitude_normal, magnitude_ortho, theta = find_maximal_blur_direction(gradient_magnitudes, n_angles=n_angles,
                                                                           n_interpolated_angles=n_interpolated_angles)
    # finally compute the Gaussian parameters
    sigma_0, sigma_1 = compute_gaussian_parameters(magnitude_normal, magnitude_ortho, c=c, b=b)
    # create the blur kernel
    kernel = create_gaussian_filter(sigma_0, sigma_1, theta, ksize=ker_size)
    return kernel


def normalize(img, q):
    value_min = np.quantile(img, q=q, axis=(0, 1))
    value_max = np.quantile(img, q=1-q, axis=(0, 1))
    img_normalized = (img - value_min) / (value_max - value_min)
    return np.clip(img_normalized, 0.0, 1.0)


def compute_gradients(img):
    gradient_x, gradient_y = fourier_gradients(img)
    return gradient_x, gradient_y


def compute_gradient_magnitudes(gradients, n_angles=6):
    gradient_x, gradient_y = gradients
    gradient_magnitudes = np.zeros(n_angles+1)
    for i in range(n_angles+1):
        angle = i * np.pi / n_angles
        gradient_at_angle = gradient_x * np.cos(angle) - gradient_y * np.sin(angle)
        gradient_magnitude_at_angle = np.amax(np.abs(gradient_at_angle))
        gradient_magnitudes[i] = gradient_magnitude_at_angle
    return gradient_magnitudes


def find_maximal_blur_direction(gradient_magnitudes, n_angles=6, n_interpolated_angles=30):
    # first build all sampled orientations
    thetas = np.linspace(0, 180, n_angles+1)
    # interpolate at new angles the magnitude
    interpolated_thetas = np.array([i*180.0/n_interpolated_angles for i in range(n_interpolated_angles)])
    interpolator = interpolate.interp1d(thetas, gradient_magnitudes, kind='cubic')
    interpolated_gradient_magnitudes = interpolator(interpolated_thetas)
    i_min = np.argmin(interpolated_gradient_magnitudes)
    theta_normal = interpolated_thetas[i_min]
    magnitude_normal = interpolated_gradient_magnitudes[i_min]
    # get orthogonal magnitude
    theta_ortho = (theta_normal + 90.0) % 180  # angle in [0,pi)
    i_ortho = int(theta_ortho // (180 / n_interpolated_angles))
    magnitude_ortho = interpolated_gradient_magnitudes[i_ortho]
    return magnitude_normal, magnitude_ortho, theta_normal * np.pi / 180


def compute_gaussian_parameters(magnitude_normal, magnitude_ortho, c=89.8, b=0.764):
    sigma_0 = np.sqrt(np.maximum(c**2/magnitude_normal**2 - b**2, 1e-8))
    sigma_0 = np.clip(sigma_0, 0.3, 4.0)
    sigma_1 = np.sqrt(np.maximum(c**2/magnitude_ortho**2 - b**2, 1e-8))
    sigma_1 = np.clip(sigma_1, 0.3, 4.0)
    return sigma_0, sigma_1


def create_gaussian_filter(sigma_0, sigma_1, theta, ksize=25):
    sigma = (sigma_0, sigma_1)
    ksize = np.array([ksize, ksize])
    kernel = gaussian_filter(sigma=sigma, theta=theta, k_size=ksize)
    return kernel





##############################################
################# Pytorch ####################
##############################################



class GaussianBlurEstimator(nn.Module):
    def __init__(self, n_angles, n_interpolated_angles, c, b, k_size):
        super(GaussianBlurEstimator, self).__init__()
        self.n_angles = n_angles
        self.n_interpolated_angles = n_interpolated_angles
        self.c = c
        self.b = b
        self.k_size = k_size

    def forward(self, images):
        images_norm = self._normalize(images)
        gradients = fourier_gradients(images_norm)
        gradient_magnitudes_angles = self._compute_magnitudes_at_angles(gradients)
        magnitudes_normal, magnitudes_ortho, thetas = self._find_direction(images_norm, gradient_magnitudes_angles)
        sigmas, rhos = self._find_variances(magnitudes_normal, magnitudes_ortho)
        kernels = self._create_gaussian_filter(thetas, sigmas, rhos).to(images.device)
        return kernels, (thetas, sigmas, rhos)

    def _normalize(self, images, q=0.0001):
        value_min = np.quantile(images, q=q, axis=(-2, -1), keepdims=True).astype(np.float32)
        value_max = np.quantile(images, q=1 - q, axis=(-2, -1), keepdims=True).astype(np.float32)
        images = (images - value_min) / (value_max - value_min)
        return images.clamp(0.0, 1.0)

    def _compute_magnitudes_at_angles(self, gradients):
        n_angles = self.n_angles
        gradient_x, gradient_y = gradients  # (B,C,H,W)
        gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
        gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
        angles = torch.linspace(0, np.pi, n_angles + 1, device=gradient_x.device).view(1, -1, 1, 1, 1)  # (1,N,1,1,1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        gradient_magnitudes_angles = (cos * gradient_x_gray - sin * gradient_y_gray).abs()  # (B,N,1,H,W)
        gradient_magnitudes_angles = torch.amax(gradient_magnitudes_angles, dim=(-3, -2, -1))  # (B,N)
        return gradient_magnitudes_angles

    def _find_direction(self, images, gradient_magnitudes_angles):
        ## Find thetas
        thetas = np.linspace(0, 180, self.n_angles+1, dtype=np.float32)
        interpolated_thetas = np.array([i * 180.0 / n_interpolated_angles for i in range(n_interpolated_angles)],
                                       dtype=np.float32)
        interpolator = interpolate.interp1d(thetas, gradient_magnitudes_angles.cpu().numpy(), kind='cubic', axis=-1)
        gradient_magnitudes_interpolated_angles = interpolator(interpolated_thetas).astype(np.float32)
        # gradient_magnitudes_interpolated_angles = F.interpolate(gradient_magnitudes_angles.unsqueeze(1).unsqueeze(1),
        #                                                         size=(1, self.n_interpolated_angles), mode='bicubic',
        #                                                         align_corners=True).squeeze(1).squeeze(1).cpu().numpy()
        ## Compute magnitude in theta
        i_min = np.argmin(gradient_magnitudes_interpolated_angles, axis=1)
        thetas_normal = interpolated_thetas[i_min]
        magnitudes_normal = [gradient_magnitudes_interpolated_angles[i, i_min[i]] for i in range(images.shape[0])]
        magnitudes_normal = torch.tensor(magnitudes_normal).to(images.device)  # (B)
        ## Compute magnitude in theta+90
        thetas_ortho = (thetas_normal + 90.0) % 180  # angle in [0,pi)
        i_ortho = (thetas_ortho // (180 / self.n_interpolated_angles)).astype(np.int32)
        magnitudes_ortho = [gradient_magnitudes_interpolated_angles[i, i_ortho[i]] for i in range(images.shape[0])]
        magnitudes_ortho = torch.tensor(magnitudes_ortho).to(images.device)  # (B)
        return magnitudes_normal[:, None], magnitudes_ortho[:, None], torch.tensor(thetas_normal)[:, None] * np.pi / 180

    def _find_variances(self, magnitudes_normal, magnitudes_ortho):
        a = self.c**2
        b = self.b**2
        ## Compute sigma
        sigma = a / (magnitudes_normal ** 2 + 1e-8) - b
        sigma = torch.sqrt(sigma).clamp(0.3, 4.0)
        ## Compute rho
        rho = a / (magnitudes_ortho ** 2 + 1e-8) - b
        rho = torch.sqrt(rho).clamp(0.3, 4.0)
        return sigma, rho

    def _create_gaussian_filter(self, thetas, sigmas, rhos):
        k_size = self.k_size
        B = len(sigmas)
        C = 1
        # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
        lambda_1 = sigmas
        lambda_2 = rhos
        thetas = -thetas

        # Set COV matrix using Lambdas and Theta
        LAMBDA = torch.zeros(B, C, 2, 2)  # (B,C,2,2)
        LAMBDA[:, :, 0, 0] = lambda_1**2
        LAMBDA[:, :, 1, 1] = lambda_2**2
        Q = torch.zeros(B, C, 2, 2)  # (B,C,2,2)
        Q[:, :, 0, 0] = torch.cos(thetas)
        Q[:, :, 0, 1] = -torch.sin(thetas)
        Q[:, :, 1, 0] = torch.sin(thetas)
        Q[:, :, 1, 1] = torch.cos(thetas)
        SIGMA = torch.einsum("bcij,bcjk,bckl->bcil", [Q, LAMBDA, Q.transpose(-2, -1)])  # (B,C,2,2)
        INV_SIGMA = torch.linalg.inv(SIGMA)
        INV_SIGMA = INV_SIGMA.view(B, C, 1, 1, 2, 2)  # (B,C,1,1,2,2)

        # Set expectation position
        MU = (k_size//2) * torch.ones(B, C, 2)
        MU = MU.view(B, C, 1, 1, 2, 1)  # (B,C,1,1,2,1)

        # Create meshgrid for Gaussian
        X, Y = torch.meshgrid(torch.arange(k_size),
                              torch.arange(k_size),
                              indexing='xy')
        Z = torch.stack([X, Y], dim=-1).unsqueeze(-1)  # (k,k,2,1)

        # Calculate Gaussian for every pixel of the kernel
        ZZ = Z - MU
        ZZ_t = ZZ.transpose(-2, -1)  # (B,C,k,k,1,2)
        raw_kernels = torch.exp(-0.5 * (ZZ_t @ INV_SIGMA @ ZZ).squeeze(-1).squeeze(-1))  # (B,C,k,k)

        # Normalize the kernel and return
        mask_small = torch.sum(raw_kernels, dim=(-2, -1)) < 1e-2
        if mask_small.any():
            raw_kernels[mask_small].copy_(0)
            raw_kernels[mask_small, k_size//2, k_size//2].copy_(1)
        kernels = raw_kernels / torch.sum(raw_kernels, dim=(-2, -1), keepdim=True)
        return kernels


if __name__ == '__main__':
    from skimage import data, img_as_float32
    import matplotlib.pyplot as plt
    from scipy import ndimage

    img = img_as_float32(data.camera())
    # img = img_as_float32(data.astronaut())
    if img.ndim == 2:
        # imblur = ndimage.gaussian_filter(img, sigma=(0.5, 3.0), mode='wrap')
        imblur = ndimage.gaussian_filter(img, sigma=(3.0, 0.5), mode='wrap')
    else:
        imblur = np.zeros_like(img)
        for c in range(3):
            imblur[..., c] = ndimage.gaussian_filter(img[..., c], sigma=(0.5, 3.0), mode='wrap')
    np.random.seed(0)
    imblur = np.clip(imblur + 0.01 * np.random.randn(*imblur.shape), 0.0, 1.0)

    # blur estimation options
    c = 0.362
    b = 0.468
    ker_size = 31
    n_angles = 6
    n_interpolated_angles = 30

    ## Numpy estimate
    kernel_np = gaussian_blur_estimation(imblur, c=c, b=b, ker_size=ker_size)

    ## Pytorch estimate
    gaussian_estimator = GaussianBlurEstimator(n_angles=n_angles, n_interpolated_angles=n_interpolated_angles,
                                                k_size=ker_size, c=c, b=b)
    imblur_th = utils.to_tensor(imblur).unsqueeze(0).float()
    imblur_th = torch.cat([imblur_th, imblur_th.flip(-2), imblur_th.rot90(k=1, dims=(-2,-1))], dim=0)
    kernel_th, _ = gaussian_estimator(imblur_th)
    kernel_th = kernel_th[0:1]

    kernel_th = utils.to_array(kernel_th.squeeze(0))

    plt.figure()
    plt.imshow(kernel_th / kernel_th.max())
    plt.title('Torch')
    plt.show()

    plt.figure()
    plt.imshow(kernel_np / kernel_np.max())
    plt.title('Numpy')
    plt.show()

    print('Diff Numpy/Pytorch: %2.5f' % np.linalg.norm(kernel_np.astype(np.float32) -kernel_th))

    print('done')
