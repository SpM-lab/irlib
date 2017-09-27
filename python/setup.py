from setuptools import setup, find_packages, Extension
import numpy
from numpy.distutils.system_info import get_info

lapack_opt = numpy.__config__.lapack_opt_info

if not lapack_opt:
    raise NotFoundError('no lapack/blas resources found')

extra_link_args = lapack_opt.get('extra_link_args', [])
extra_compile_args = lapack_opt.get('extra_compile_args', [])

setup(
    name = "irlib",
    version = "0.1",
    author='Hiroshi Shinaoka, Naoya Chikano, Junya Otsuki',
    author_email='h.shinaoka@gmail.com',
    description='Python binding for irlib',
    packages = ['irlib'],
    setup_requires=["numpy", "scipy"],
    install_requires=["numpy", "scipy"],
    include_dirs=[numpy.get_include()],
    ext_modules=[
        Extension('_basis',
        sources=['./basisPYTHON_wrap.cxx'],
        include_dirs=['../include', '../thirdparty/eigen3', '../thirdparty/mpfr_cxx'],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        #library_dirs=['/'],
        #libraries = ['blas']
        #extra_compile_args=['']
        )
   ],
)
