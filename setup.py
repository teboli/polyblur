from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension
import re

# # Install the intermediate separable kernel
# setup(name='separable_gaussian2d',
#         ext_modules=[cpp_extension.CppExtension('separable_gaussian2d', ['./polyblur/separable_convolution/separable_gaussian2d.cpp'])],
#         cmdclass={'build_ext': cpp_extension.BuildExtension})


with open("long_description.txt", "r") as fh:
    long_description = fh.read()


# Install the main polyblur module
setup(
    name='polyblur',
    version="1.0.1",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    description="Breaking down Polyblur: Fast blind Correction of Small Anisotropic Blurs [IPOL2022]",
    url="https://github.com/teboli/polyblur",
    long_description=long_description,
    packages = find_packages(),
    include_package_data=True,
    install_requires=["torch-tools"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
