# tuning
Python scripts for tuning curve optimization according to Efficient Coding Hypotheses 

Note: currently support python3

## Installation instructions For Mac OS


### Step 1. Check whether python or python3 is installed 
(terminal commands: `which python`, `which python3`) 


### Step 2. Install Anaconda3 and Cython:
- Download installer of Anaconda3:  https://docs.anaconda.com
- Set the path:
      ```export PATH=~/anaconda3/bin:$PATH```
- Install Cython:
      ```conda install -c anaconda cython```
- For convinience of programming, Jupiter Notebook is also recommended: https://jupyter.readthedocs.io/en/latest/install.html
 

### Step 3. In the folder of the codes, build the cython programs:

```python setup_cyMIPoisson.py build_ext --inplace```

```python setup_cyMIBN.py build_ext --inplace```

If it shows up an error : 
clang: error: unsupported option '-fopenmp'
error: command 'gcc' failed with exit status 1

This means that OpenMp is missing.
Solution:
 - install a new version of gcc with home-brew:
        ```brew install gcc```
 - install the llvm and libomp packages:
	```brew install llvm libomp```
 - link default complier to gcc: for example, with gcc-8 version:
        ```export CC=gcc-8```
        ```export CXX=g++-8```
Reference: https://stackoverflow.com/questions/36211018/clang-error-errorunsupported-option-fopenmp-on-mac-osx-el-capitan-buildin



### Step 4. Probably also need to install ffmeg [for plotting animations]:
```brew install ffmpeg```


### Step 5. Try to run example.py :
```python example.py```

Or:

```python3 example.py```


