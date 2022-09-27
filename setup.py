from setuptools import setup, Extension
from torch.utils import cpp_extension


# Install the intermediate domain transform module
setup(name='fast_domain_transform',
        ext_modules=[cpp_extension.CppExtension('fast_domain_transform', ['./polyblur/domain_transform/RF.cpp'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

# Install the main polyblur module
setup(
    name='polyblur',
    version="1.0.0",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    description="Breaking down Polyblur: Fast blind Correction of Small Anisotropic Blurs [IPOL2022]",
    url="https://github.com/teboli/polyblur",
    packages = setuptools.find_packages(),
    include_package_data=True,
)
