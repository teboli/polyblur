import os
from skimage import img_as_float32, img_as_ubyte, io
import matplotlib.pyplot as plt
import deblurring
import utils


def main():
    ## Synthetic
    imblur = img_as_float32(plt.imread('peacock_defocus.png'))

    # blur estimation options
    c = 0.374
    b = 0.461

    # deblurring options
    patch_decomposition = True
    patch_size = 400
    patch_overlap = 0.25
    batch_size = 20
    n_iter = 3
    alpha = 6
    beta = 1
    masking = True
    edgetaping = True
    prefiltering = True

    # blind deblurring
    deblurrer = deblurring.Polyblur(patch_decomposition=patch_decomposition, patch_size=patch_size,
                                    patch_overlap=patch_overlap, batch_size=batch_size)

    imblur = utils.to_tensor(imblur).unsqueeze(0)
    impred = deblurrer(imblur, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, masking=masking, edgetaping=edgetaping,
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

    io.imsave(os.path.join('results/peacock_restored_alpha_%d_beta_%d.png' % (alpha, beta)), img_as_ubyte(impred))

    print('done')


if __name__ == '__main__':
    main()
