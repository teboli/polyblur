from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fast_domain_transform',
        ext_modules=[cpp_extension.CppExtension('fast_domain_transform', ['RF.cpp'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
