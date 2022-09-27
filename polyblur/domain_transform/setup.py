from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fast_domain_transform',
        ext_modules=[cpp_extension.CppExtension('./recursive_filter.cpp', ['recursive_filter.cpp'])],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
