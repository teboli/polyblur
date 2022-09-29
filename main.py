import os
import numpy as np
import torch
import argparse
from skimage import img_as_float32, img_as_ubyte, io, color
from scipy import ndimage
import matplotlib.pyplot as plt

from polyblur import PolyblurDeblurring
from polyblur import utils, filters
import time


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#### Argparser

def str2bool(v):
    v = str(v)

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
## Image parameter
parser.add_argument('--impath', type=str, help='input image')

## Synthetic parameters
parser.add_argument('--synthetic_degradation', type=str2bool, default=False, help='if set adds synthetic gaussian blur')
parser.add_argument('--sigma', type=float, default=3.0, help='first eigenvector of the gaussian kernel')
parser.add_argument('--rho', type=float, default=1.0, help='second eigenvector of the gaussian kernel')
parser.add_argument('--theta', type=float, default=0.0, help='angle of the gaussian kernel')
parser.add_argument('--sigma_n', type=float, default=0.01, help='image noise sigma')

## Polyblur parameters
parser.add_argument('--N', type=int, default=3, help='polyblur iterations')
parser.add_argument('--alpha', type=int, default=2, help='polyblur alpha parameter')
parser.add_argument('--beta', type=int, default=3, help='polyblur beta parameter')
parser.add_argument('--q', type=float, default=0, help='quantile value for image normalization')
parser.add_argument('--do_prefiltering', type=str2bool, default=False, help='apply noise prefiltering')
parser.add_argument('--do_halo_removal', type=str2bool, default=False, help='use halo removal correction')
parser.add_argument('--do_edgetaping', type=str2bool, default=False, help='do edgetaper to better handle edges')

## Patch parameters
parser.add_argument('--do_patch_decomposition', type=str2bool, default=False, help='process the image by patches')
parser.add_argument('--patch_size', type=int, default=400, help='size of the patches')
parser.add_argument('--patch_overlap', type=float, default=0.25, help='patch overlap')

args = parser.parse_args()


print('Polyblur runs with parameters:')
print('  synthetic_degradation:  %s' % args.synthetic_degradation)
if args.synthetic_degradation:
    print('  sigma:                  %1.1f' % args.sigma)
    print('  rho:                    %1.1f' % args.rho)
    print('  theta:                  %d' % args.theta)
    print('  sigma_n:                %2.2f' % args.sigma_n)
print('  N:                      %d' % args.N)
print('  alpha:                  %d' % args.alpha)
print('  beta:                   %d' % args.beta)
print('  do_prefiltering:        %s' % args.do_prefiltering)
print('  do_edgetaping:          %s' % args.do_edgetaping)
print('  do_halo_removal:        %s' % args.do_halo_removal)
print('  do_patch_decomposition: %s' % args.do_patch_decomposition)
if args.do_patch_decomposition:
    print('  patch_size:             %s' % args.patch_size)
    print('  patch_overlap:          %s' % args.patch_overlap)
print()



#### Read input
img = img_as_float32(io.imread(args.impath))
if img.shape[-1] == 4:
    img = color.rgba2rgb(img)

print('Processing a (%d,%d) image.' % (img.shape[1], img.shape[0]))
print()


#### (optional) Create a synthetic blurry image
if args.synthetic_degradation:
    kernel = filters.gaussian_filter((args.sigma, args.rho), theta=args.theta * np.pi/180, k_size=np.array([25, 25]))
    if img.ndim == 2:
        imblur = ndimage.convolve(img, kernel, mode='wrap')[..., None]  # (H,W,1)
    else:
        imblur = ndimage.convolve(img, kernel[..., None], mode='wrap')  # (H,W,3)
    imblur += args.sigma_n * np.random.randn(*(imblur.shape))
    imblur = np.clip(imblur, 0.0, 1.0)
else:
    imblur = img


#### Restoration
deblurrer = PolyblurDeblurring(patch_decomposition=args.do_patch_decomposition, patch_size=args.patch_size,
                               patch_overlap=args.patch_overlap, batch_size=20)

c = 0.362
b = 0.468

if torch.cuda.is_available():
    method = 'direct'
else:
    method = 'fft'

imblur = utils.to_tensor(imblur).unsqueeze(0).to(device)
start = time.time()
impred = deblurrer(imblur, n_iter=args.N, c=c, b=b, alpha=args.alpha, beta=args.beta, 
                   remove_halo=args.do_halo_removal, prefiltering=args.do_prefiltering, 
                   edgetaping=args.do_edgetaping, method=method, q=args.q)
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

io.imsave(os.path.join('results/restored_alpha_%d_beta_%d.png' % (args.alpha, args.beta)), img_as_ubyte(impred))

print('done')

