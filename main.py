import numpy as np
from skimage import data, img_as_float32
from scipy import ndimage
import matplotlib.pyplot as plt
import deblurring


def main():
    img = img_as_float32(data.camera())
    # img = img_as_float32(data.astronaut())
    if img.ndim == 2:
        imblur = ndimage.gaussian_filter(img, sigma=(0.5, 3.0), mode='wrap')
    else:
        imblur = np.zeros_like(img)
        for c in range(3):
            imblur[..., c] = ndimage.gaussian_filter(img[..., c], sigma=(0.5, 3.0), mode='wrap')
    np.random.seed(0)
    imblur = np.clip(imblur + 0.01 * np.random.randn(*imblur.shape), 0.0, 1.0)

    # blur estimation options
    c = 0.35
    sigma_b = 0.768
    # c = 0.17
    # sigma_b = 0.0

    # deblurring options
    alpha = 10
    b = 3
    masking = True

    # blind deblurring
    impred1 = deblurring.polyblur(imblur, n_iter=1, c=c, sigma_b=sigma_b, alpha=alpha, b=b, masking=masking)
    impred2 = deblurring.polyblur(imblur, n_iter=2, c=c, sigma_b=sigma_b, alpha=alpha, b=b, masking=masking)
    impred3 = deblurring.polyblur(imblur, n_iter=3, c=c, sigma_b=sigma_b, alpha=alpha, b=b, masking=masking)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(imblur, cmap='gray')
    plt.axis('off')
    plt.title('Blurry')
    plt.subplot(1, 4, 2)
    plt.imshow(impred1, cmap='gray')
    plt.axis('off')
    plt.title('Prediction 1')
    plt.subplot(1, 4, 3)
    plt.imshow(impred2, cmap='gray')
    plt.axis('off')
    plt.title('Prediction 2')
    plt.subplot(1, 4, 4)
    plt.imshow(impred3, cmap='gray')
    plt.axis('off')
    plt.title('Prediction 3')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
