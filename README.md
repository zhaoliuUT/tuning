# tuning
Python scripts for tuning curve optimization according to Efficient Coding Hypotheses 

Note: currently only works for python 3.7 or above.

## Installation instructions


### Step 1. Prerequisites:
- `python`, `Anaconda`, `pip`

	References: 
	
	Linux: https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
	
	Mac Os: https://docs.anaconda.com/anaconda/install/mac-os/

- For convenience, it is better to create a virtual environment to manage the packages and run the codes (or create a conda environment). 

	- To create a conda environment with specified python version with name `my-env`, for example, 
			
		```conda create -n my-env python=3.6```

		And to enter `my-env`,

		```conda activate my-env```
		
		To exit,
		
		```conda deactivate```
		
	- For python 3.5 or higher, you can create a virtual environment using `venv`. For example, a virtual environment with name `my-env`:
	
		```python -m venv my-env```

		And to enter `my-env`,

		```source my-env/bin/activate```
		
		Press Ctrl + D / Command + D to exit.
		
		Reference: https://docs.python.org/3/library/venv.html


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
 
### Step 2. Clone the repository locally

In your terminal, in a directory of your choice, use ```git init``` to create a new git repository. Then clone this remote repository to your local:

```git clone https://github.com/zhaoliuUT/tuning.git```

Alternatively, you can download the zip file of the repository at the top of the main page of the repository.

Reference: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository

### Step 3. Install the package locally

Run in your lcoal directory

```python setup.py install```

If it shows up an error : 

```clang: error: unsupported option '-fopenmp'
error: command 'gcc' failed with exit status 1
```

This means that OpenMp is missing.

Solution on Mac Os:
 - install a new version of gcc with home-brew:
 
	```brew install gcc```
	
 - install the llvm and libomp packages:
 
	```brew install llvm libomp```
	
 - link default complier to gcc: for example, with gcc-8 version:
 
	```export CC=gcc-8```
	
	```export CXX=g++-8```
	
	Reference: https://stackoverflow.com/questions/36211018/clang-error-errorunsupported-option-fopenmp-on-mac-osx-el-capitan-buildin


### Step 4. Try to run example.py :

```python examples/example.py```

Running this example takes about 3 mins on a laptop computer equipped with an 4 cores CPU Intel i7-6500U 2.50 GHZ.

It will print to the console 10 iterations of optimizing a Poissonian Model tuning curve, 10 iterations of optimizing a Binary Model tuning curve, and 5+5 (channel+capacity) iterations of optimizing a Noncyclic Model tuning curve.
In the subdirectory `data`, the following data files / animations can be found:

```test0_bn.mp4  test0.mp4  test0_noncyclic.mp4  test0_noncyclic_cube.mp4 ```

```test0.npy test0_bn.npy    test0_noncyclic.npy```
