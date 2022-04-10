import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from skimage import img_as_float
from scipy import ndimage, optimize, interpolate

from filters import fourier_gradients, gaussian_filter


def generate_blurry_image(img, kernel_parameters, patch_size, ker_size, noise_std):
    sigma_max, sigma_min, rho_max, rho_min = kernel_parameters
    h, w = img.shape[:2]
    # Select a random patch
    i0 = np.random.choice(h - patch_size[0] - 1)
    j0 = np.random.choice(w - patch_size[1] - 1)
    patch = img[i0:i0 + patch_size[0], j0:j0 + patch_size[1]]
    # Select random noise
    noise = np.random.randn(*patch.shape)
    # Generate blur kernel
    sigma_0_gt = float((sigma_max - sigma_min) * np.random.rand(1) + sigma_min)
    rho_gt = float((rho_max - rho_min) * np.random.rand(1) + rho_min)
    sigma_1_gt = max(0.3,sigma_0_gt * rho_gt)
    theta_gt = np.random.choice(180) * np.pi / 180
    # Generate the kernel
    kernel_gt = gaussian_filter(sigma=(sigma_0_gt, sigma_1_gt), theta=theta_gt, k_size=np.array([ker_size, ker_size]))
    # Generate blurry image
    p_blur = ndimage.convolve(patch, kernel_gt, mode='wrap')
    p_blur = normalize_np(p_blur, q=0.0001)
    p_blur = np.clip(p_blur + noise_std * noise, 0.0, 1.0)
    # p_blur = np.round(p_blur)
    # Pack parameters
    parameters_gt = (sigma_0_gt, sigma_1_gt, rho_gt, theta_gt)
    return patch, p_blur, kernel_gt, parameters_gt


def normalize_np(img, q):
    value_min = np.quantile(img, q=q, axis=(0, 1))
    value_max = np.quantile(img, q=1-q, axis=(0, 1))
    img_normalized = (img - value_min) / (value_max - value_min)
    return np.clip(img_normalized, 0.0, 1.0)


def main(image_paths, n_kernel_per_image, kernel_parameters, patch_size=(400, 400), ker_size=35):
    n_images = len(image_paths)
    indexes = list(np.arange(n_images))
    indexes_out = [0,2,4,5,6,8,9,10,13,14,23,26,27,28,30,31,37,38,40,41,42,43,44,46,50,51,52,
                   56,61,62,65,67,69,72,79,81,85,87,92,93,97]
    for i in indexes_out[::-1]:
        del indexes[i]

    noise_std = 0.01

    n_images = len(indexes)
    print('n_images:', n_images)
    n_samples = n_images * n_kernel_per_image
    sampled_values_n_rgb = np.zeros((n_samples, 2))
    sampled_values_o_rgb = np.zeros((n_samples, 2))

    p = 0
    for i, image_path in tqdm(enumerate(image_paths)):
        if i in indexes:
            # Read the image
            img = img_as_float(plt.imread(image_path))
            h, w = img.shape[:2]
            img = img[h//4:-h//4, w//4:-w//4, 1]
            for j in range(n_kernel_per_image):
                n_sample = p * n_kernel_per_image + j
                np.random.seed(n_sample)

                ### Generate blurry image and patch
                p_sharp, p_blur, kernel_gt, parameters_gt = generate_blurry_image(img, kernel_parameters,
                                                                                  patch_size, ker_size, noise_std)
                sigma_0_gt, sigma_1_gt, rho_gt, theta_gt = parameters_gt

                ### Compute gradients - RGB image
                g_x, g_y = fourier_gradients(p_blur)
                thetas = np.linspace(0, np.pi, 7)
                g = [g_x * np.cos(theta) - g_y * np.sin(theta) for theta in thetas]
                a = np.array([np.amax(np.abs(gg)) for gg in g])
                f = interpolate.interp1d(thetas, a, kind='cubic')
                ag = f(np.arange(180) * np.pi / 180)
                i_n = np.argmin(ag)
                f_n = np.amax(ag[i_n])
                f_o = np.amax(ag[(i_n + 90) % 180])
                sampled_values_n_rgb[n_sample, 0] = 1. / (f_n + 1e-8)
                sampled_values_n_rgb[n_sample, 1] = sigma_0_gt
                sampled_values_o_rgb[n_sample, 0] = 1. / (f_o + 1e-8)
                sampled_values_o_rgb[n_sample, 1] = sigma_1_gt
            p += 1

    ### Evaluate and plot rgb data
    print('Optimizing')
    X_n_rgb = sampled_values_n_rgb[:, 0]
    Y_n_rgb = sampled_values_n_rgb[:, 1]
    c_n_rgb, s_n_rgb, c2_n_rgb, s2_n_rgb = do_optimization(X_n_rgb**2, Y_n_rgb**2)

    X_o_rgb = sampled_values_o_rgb[:, 0]
    Y_o_rgb = sampled_values_o_rgb[:, 1]
    c_o_rgb, s_o_rgb, c2_o_rgb, s2_o_rgb = do_optimization(X_o_rgb**2, Y_o_rgb**2)

    markersize = 2.0
    plt.figure(figsize=(8, 6))
    add_plot(c2_n_rgb, s2_n_rgb, X_n_rgb**2, Y_n_rgb**2, xlabel=r'$1/\|\nabla_\theta n(v)\|_\infty^2$',
             ylabel=r'$\sigma^2$')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    add_plot(c2_o_rgb, s2_o_rgb, X_o_rgb ** 2, Y_o_rgb ** 2, xlabel=r'$1/\|\nabla_{\theta+\frac{\pi}{2}} n(v)\|_\infty^2$',
             ylabel=r'$\rho^2$')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    add_plot(c2_n_rgb, s2_n_rgb, X_n_rgb, Y_n_rgb, xlabel=r'$1/\|\nabla_\theta n(v)\|_\infty$', ylabel=r'$\sigma$',
             do_sqrt=True)
    # add_plot(c_n_rgb, s_n_rgb, X_n_rgb, Y_n_rgb, xlabel=r'$1/\|\nabla_\theta n(v)\|_\infty$', ylabel=r'$\sigma$',
    #          do_sqrt=True)
    plt.tight_layout()
    plt.savefig('./results/calibration_normal_%2.2f.jpg' % noise_std)
    plt.show()

    plt.figure(figsize=(8, 6))
    add_plot(c2_o_rgb, s2_o_rgb, X_o_rgb, Y_o_rgb, xlabel=r'$1/\|\nabla_{\theta+\frac{\pi}{2}} n(v)\|_\infty$',
             ylabel=r'$\rho$', do_sqrt=True)
    # add_plot(c_o_rgb, s_o_rgb, X_o_rgb, Y_o_rgb, xlabel=r'$1/\|\nabla_{\theta+\frac{\pi}{2}} n(v)\|_\infty$',
    #          ylabel=r'$\rho$', do_sqrt=True)
    plt.tight_layout()
    plt.savefig('./results/calibration_orthogonal_%2.2f.jpg' % noise_std)
    plt.show()


    print('Done!')


def MAE(a, b, x, y):
    return np.mean(np.abs(a**2 * x - b**2 - y))


def optimizeMAE(x, y):
    d = len(x)
    n = 2
    c = np.zeros(d+n)
    c[0:d] = 1
    X = np.stack([x, np.ones(d)], axis=-1)
    I = np.eye(d)
    A1 = np.concatenate([-I, X], axis=-1)
    A2 = np.concatenate([-I, -X], axis=-1)
    A = np.concatenate([A1, A2], axis=0)
    b = np.concatenate([y, -y], axis=0)
    bounds = [(0, None) for _ in range(d)] + [(None, None), (None, 0)]
    res = optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, options={'disp': True, 'tol': 1e-6})
    return res.x[-2], res.x[-1]


def do_optimization(x, y):
    c2_normal, s2_normal = optimizeMAE(x, y)
    print('(c^2, sigma_b^2) = (%2.3f, %2.3f)' % (c2_normal, s2_normal))
    c_normal, s_normal = np.sqrt(c2_normal), np.sign(s2_normal) * np.sqrt(np.abs(s2_normal))
    print('(c, sigma_b) = (%2.3f, %2.3f)' % (c_normal, s_normal))
    return c_normal, s_normal, c2_normal, s2_normal


def add_plot(a, b, x, y, xlabel, ylabel, do_sqrt=False):
    def regressed_function(value, max_value, min_value, slope, origin):
        return np.maximum(np.minimum(slope * value + origin, max_value), min_value)

    fontsize_label = 22
    fontsize_ticks = 14
    linewidth = 2.5
    markersize = 4

    g_min = x.min()
    g_max = x.max()
    n_samples = len(x)
    value_range = np.linspace(g_min, g_max, n_samples)
    value_range = np.array(sorted(value_range))
    s_max = 4.0
    s_min = 0.3
    if do_sqrt:
        # aa = a**2
        # bb = np.sign(b) * b**2
        aa = a
        bb = b
        regressed_values = np.sqrt(regressed_function(value_range**2, s_max**2, s_min**2, aa, bb))
    else:
        regressed_values = regressed_function(value_range, s_max**2, s_min**2, a, b)
    plt.plot(x, y, 'o', color='b', markersize=markersize)
    plt.plot(value_range, regressed_values, color='r', linewidth=linewidth)
    plt.xlabel(xlabel, fontsize=fontsize_label)
    plt.ylabel(ylabel, fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True)
    # plt.title('Slope = %2.3f | Intercept = %2.3f' % (a, b), fontsize=fontsize_label)


if __name__ == '__main__':
    sns.set(font_scale=1.5, rc={'text.usetex': True})
    sns.set_style("whitegrid")

    n_kernel_per_image = 10
    sigma_max = 4.0
    sigma_min = 0.3
    rho_max = 1.0
    rho_min = 0.33
    kernel_parameters = (sigma_max, sigma_min, rho_max, rho_min)

    path_to_images = '/Users/thomas/Documents/INRIA/blind_PSF_estimation/pictures/DIV2K_valid_HR/*.png'
    image_paths = sorted(glob(path_to_images))[:100]

    main(image_paths, n_kernel_per_image, kernel_parameters)
