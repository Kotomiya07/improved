from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.chdir('score_sde/op/')

setup(
    name='upfirdn2d_op',
    ext_modules=[
        CUDAExtension(
            name='upfirdn2d_op', 
            sources=['upfirdn2d.cpp', 'upfirdn2d_kernel.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
        }
)

setup(
    name='fused',
    ext_modules=[
        CUDAExtension(
            name='fused', 
            sources=['fused_bias_act.cpp', 'fused_bias_act_kernel.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
        }
)
