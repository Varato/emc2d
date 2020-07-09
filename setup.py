from setuptools import setup, find_packages, Extension
import numpy as np

extensions = Extension("emc2d.pykernels",
                        sources=[],
                        include_dir=[np.get_include()])

setup(
    name="emc2d",
    version='0.0.0',
    packages=find_packages(),
    ext_modules = []

)