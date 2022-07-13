import os
import torch
from skimage import img_as_float32, img_as_ubyte, io
import matplotlib.pyplot as plt
import deblurring
import utils
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    ## Synthetic
    imblur = img_as_float32(plt.imread('peacock_defocus.png'))

    # blur estimation options
    c = 0.362
    b = 0.468

    # deblurring options
    patch_decomposition = False 
    patch_size = 400
    patch_overlap = 0.25
    batch_size = 20
    n_iter = 3
    alpha = 6
    beta = 1
    masking = True
    edgetaping = False
    prefiltering = True

    # blind deblurring
    deblurrer = deblurring.Polyblur(patch_decomposition=patch_decomposition, patch_size=patch_size,
                                    patch_overlap=patch_overlap, batch_size=batch_size)

    imblur = utils.to_tensor(imblur).unsqueeze(0).to(device)
    start = time.time()
    impred = deblurrer(imblur, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, masking=masking, edgetaping=edgetaping,
                        prefiltering=prefiltering)
    print('Restoration took %2.4f seconds' % (time.time() - start))
    imblur = utils.to_array(imblur.squeeze(0).cpu())
    impred = utils.to_array(impred.squeeze(0).cpu())

    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(imblur, cmap='gray')
    plt.axis('off')
    plt.title('Blurry')
    plt.subplot(1, 2, 2)
    plt.imshow(impred, cmap='gray')
    plt.axis('off')
    plt.title('Prediction')
    plt.tight_layout()
    plt.show()

    io.imsave(os.path.join('results/peacock_restored_alpha_%d_beta_%d.png' % (alpha, beta)), img_as_ubyte(impred))

    print('done')


if __name__ == '__main__':
    main()
