import os
import torch
from skimage import img_as_float32, img_as_ubyte, io
import matplotlib.pyplot as plt
from polyblur import PolyblurDeblurring
from polyblur import utils
import time

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def main():
    print('Will run on device: %s' % device)

    ## Synthetic
    imblur = img_as_float32(plt.imread('./pictures/peacock_defocus.png'))

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
    remove_halo = True
    edgetaping = False
    prefiltering = True
    multichannel_kernel = False
    discard_saturation = False

    # blind deblurring
    deblurrer = PolyblurDeblurring(patch_decomposition=patch_decomposition, patch_size=patch_size,
                                   patch_overlap=patch_overlap, batch_size=batch_size)

    imblur = utils.to_tensor(imblur).unsqueeze(0).to(device)
    start = time.time()
    impred = deblurrer(imblur, n_iter=n_iter, c=c, b=b, alpha=alpha, beta=beta, remove_halo=remove_halo, edgetaping=edgetaping,
                        prefiltering=prefiltering, multichannel_kernel=multichannel_kernel, discard_saturation=discard_saturation)
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
