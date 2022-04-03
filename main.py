import os
import numpy as np
from skimage import data, img_as_float32, img_as_ubyte, io, transform, filters
from scipy import ndimage
import matplotlib.pyplot as plt
import deblurring
import utils

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def main():
    ## Synthetic
    img = img_as_float32(data.camera()); name = 'camera'
    # img = img_as_float32(data.astronaut()); name = 'astronaut'
    # img = img_as_float32(data.chelsea()); name = 'chelsea'

    if img.ndim == 2:
        imblur = ndimage.gaussian_filter(img, sigma=(1.0, 2.8), mode='wrap')
    else:
        imblur = np.zeros_like(img)
        for c in range(3):
            imblur[..., c] = ndimage.gaussian_filter(img[..., c], sigma=(0.5, 3.0), mode='wrap')
    np.random.seed(0)
    imblur = np.clip(imblur + 0.01 * np.random.randn(*imblur.shape), 0.0, 1.0)

    # blur estimation options
    c = 0.362
    b = 0.468

    # deblurring options
    n_iter = 3
    alpha = 6
    beta = 1
    masking = True
    edgetaping = True
    prefiltering = True

    # blind deblurring
    imblur = utils.to_tensor(imblur).unsqueeze(0)
    impred = deblurring.polyblur(imblur, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, masking=masking, edgetaping=edgetaping,
                        prefiltering=prefiltering)

    imblur = utils.to_array(imblur.squeeze(0))
    impred = utils.to_array(impred.squeeze(0))

    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.imshow(imblur, cmap='gray')
    plt.axis('off')
    plt.title('Blurry')
    plt.subplot(2, 1, 2)
    plt.imshow(impred, cmap='gray')
    plt.axis('off')
    plt.title('Prediction')
    plt.tight_layout()
    plt.show()

    io.imsave(os.path.join('results', name + '_iter.png'), img_as_ubyte(impred))

    print('done')


if __name__ == '__main__':
    main()
