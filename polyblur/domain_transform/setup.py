from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension


sources = ['./polyblur/domain_transform/RF.cpp']  # simply compile the recursive filter code.
# sources = ['./polyblur/domain_transform/NC.cpp']
# sources = ['./polyblur/domain_transform/NC.cpp', './polyblur/domain_transform/RF.cpp']
extra_compile_args = ['-g', '-O3']

# Install the intermediate domain transform module
setup(name='domain_transform',
        ext_modules=[cpp_extension.CppExtension(name='domain_transform', 
                                                sources=sources, 
                                                extra_compile_args=extra_compile_args)],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
