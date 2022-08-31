import setuptools


setuptools.setup(
    name='polyblur',
    version="1.0.0",
    author="Thomas Eboli",
    author_email="thomas.eboli@ens-paris-saclay.fr",
    description="Breaking down Polyblur: Fast blind Correction of Small Anisotropic Blurs [IPOL2022]",
    url="https://github.com/teboli/polyblur",
    packages = setuptools.find_packages(),
    include_package_data=True,
)
