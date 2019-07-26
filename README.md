# tuning
Python scripts for tuning curve optimization according to Efficient Coding Hypotheses 

Note: currently only works for building in python2 environment; need to fix setup errors for python3 (mainly cython modules).

## Installation instructions


### Step 1. Prerequisites:
- `python`, `Anaconda`, `pip`

	References: 
	
	Linux: https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
	
	Mac Os: https://docs.anaconda.com/anaconda/install/mac-os/

- For convenience, it is better to create a virtual environment to manage the packages and run the codes (or create a conda environment). 

	- To create a python virtual environment with name `my-env`, for example, 

		```python -m venv my-env```

		And to enter `my-env`,

		```source my-env/bin/activate```

	- To create a conda environment with specified python version with name `my-env`, for example, 
			
		```conda create -n my-env python=3.6```

		And to enter `my-env`,

		```conda activate my-env```
		
		To exit,
		
		```conda deactivate```

- Python packages:
	 `numpy`, `scipy`, `matplotlib`, `Cython`

	 ```pip install numpy scipy matplotlib Cython```

	 Or:

	 ```conda install numpy scipy matplotlib```

	 ```conda install -c anaconda cython```
- To generate animations of lists of tuning curves, `ffmpeg` is also needed:

	```sudo apt install ffmpeg``` on Linux, 
	
	```brew install ffmpeg``` on MacOS.
	
	Reference: https://linuxize.com/post/how-to-install-ffmpeg-on-ubuntu-18-04/
- For convenience of visualization, Jupiter Notebook is also recommended: 
https://jupyter.readthedocs.io/en/latest/install.html
 

### Step 2. In the directory of the codes, install the package:

```python setup.py install```

If it shows up an error : 

```clang: error: unsupported option '-fopenmp'
error: command 'gcc' failed with exit status 1
```

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


### Step 3. Try to run example.py :

```python examples/example.py```

Running this example takes about 3 mins on a laptop computer equipped with an 4 cores CPU Intel i7-6500U 2.50 GHZ.

It will print to the console 10 iterations of optimizing a Poissonian Model tuning curve, 10 iterations of optimizing a Binary Model tuning curve, and 5+5 (channel+capacity) iterations of optimizing a Noncyclic Model tuning curve.
In the subdirectory `data`, the following data files / animations can be found:

```test0_bn.mp4  test0.mp4  test0_noncyclic.mp4  test0_noncyclic_cube.mp4 ```

```test0.npy test0_bn.npy    test0_noncyclic.npy```
