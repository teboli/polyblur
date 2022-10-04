import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from skimage import img_as_float32


def to_tensor(x, to_type=torch.float):
    """Converts an nd.array into a torch.tensor of size (C,H,W)."""
    ndim = len(x.shape)
    if ndim == 2:
        x = torch.from_numpy(x.copy()).unsqueeze(0)
    else:
        x = torch.from_numpy(x.copy()).permute(2, 0, 1)
    if to_type == torch.float:
        x = x.float()
    elif to_type == torch.double:
        x = x.double()
    else:
        x = x.long()
    return x


def to_array(x):
    """Converts a torch.tensor into an nd.array of size (H,W,C)."""
    x = x.squeeze().detach().cpu()
    if len(x.shape) == 2:
        return x.numpy()
    else:
        x = x.permute(1, 2, 0)
        return x.numpy()
    
    
def to_float(img):
    """Converts an ndarray to np.float32 type."""
    img = img_as_float32(img)
    img = img.astype(np.float32)
    return img


def to_uint(img):
    """Converts an ndarray to np.uint8 type."""
    img = img_as_float32(img)
    img = (255*img).astype(np.uint8)
    return img


def pad_with_kernel(img, kernel=None, ksize=3, mode='replicate'):
    if kernel is not None:
        ks = kernel.shape[-1] // 2
    else:
        ks = ksize // 2
    return F.pad(img, (ks, ks, ks, ks), mode=mode)


def crop_with_kernel(img, kernel=None, ksize=3):
    if kernel is not None:
        ks = kernel.shape[-1] // 2
    else:
        ks = ksize // 2
    return img[..., ks:-ks, ks:-ks]


def extract_tiles(img, kernel_size, stride=1):
    b, c, _, _ = img.shape
    h, w = kernel_size
    tiles = F.unfold(img, kernel_size, stride)  # (B,C*H*W,L)
    tiles = tiles.permute(0, 2, 1)  # (B,L,C*H*W)
    tiles = tiles.view(b, -1, c, h ,w)
    return tiles
    
