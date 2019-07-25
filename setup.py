#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os

use_openmp = True
compile_args = ["-Wno-unused-function", "-Wno-maybe-uninitialized",
                "-ffast-math", "-std=c++0x"]

link_args = []

if use_openmp:
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

cython_module_names = ["cyMIPoisson", "cyMIBN", "cyMINoncyclic"]

cython_modules = [Extension(name,
                  [os.path.join("./", name+".pyx")],
                  language="c++",
                  extra_compile_args = compile_args,
                  extra_link_args = link_args,
                  library_dirs = ['./'])
                  for name in cython_module_names]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tuning-opt",
    version="0.0.1",
    author="Zhao Liu",
    author_email="zliu@math.utexas.edu",
    description="Python scripts for tuning curve optimization according to Efficient Coding Hypotheses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhaoliuUT/tuning.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules = cythonize(cython_modules, compiler_directives={'language_level' : "3"}),
)