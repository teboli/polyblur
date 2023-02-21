from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

long_description = '''# Polyblur Python package

Pytorch-based implementation of "Polyblur: Removing mild blur by polynomial reblurring" [Delbracio2021] and detailed in "Breaking down Polyblur: Fast blind correction of small anisotropic blurs" [Eboli2022]. Any question at thomas.eboli@ens-paris-saclay.fr.

Import the package as

```shell
import polyblur
```

In your Python script, run the functional interface

```shell
polyblur.polyblur_deblurring
```

or the torch.nn.Module interface

```shell
polyblur.PolyblurDeblurring
```
'''


# Install the main polyblur module
setup(
    name='polyblur',
    version="0.1.5",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    description="Breaking down Polyblur: Fast blind Correction of Small Anisotropic Blurs [IPOL2022]",
    url="https://github.com/teboli/polyblur",
    long_description=long_description,
    packages = find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)


