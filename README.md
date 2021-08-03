# former

Simple transformer implementation from scratch in pytorch. See http://peterbloem.nl/blog/transformers for an in-depth 
explanation.

# Limitations

The current models are designed to show the simplicity of transformer models and self-attention. As such 
they will not scale as far as the bigger transformers. For that you'll need a number of tricks that 
complicate the code (see the blog post for details).

All models so far are a single stack of transformer blocks (that is, no encoder/decoder structures). It 
turns out that this simple configuration often works best. 

# Use

First, download or clone the repository. Then, in the directory that contains setup.py, run

```
pip install -e . 
```

The switch `-e` means when you edit the code, this changes the installed package ads well. 

Then, from the same directory, run:

```
python experiments/classify.py
```
This will run a simple classification experiment on the IMDb dataset.

Hyperparameters are passed as command line arguments. The defaults should work well. The classification data is 
automatically downloaded, and the wikipedia data is included in the repository.

You should be able to install directly from github as a package as well, with 
```
pip install git+https://github.com/pbloem/former
```
but I haven't tried this. It's probably easier to just copy over the code you need.  

## Requirements

Python 3.6+ is required.

The following should install all requirements 
```pip install torch tb-nightly tqdm numpy torchtext```

You may also need
```pip install future```
depending on the exact python version.

### conda environment

The file ```environment.yml``` describes a complete conda environment with all dependencies. After cloning or downloading the project, you create the environment as follows:

```
conda env create -f environment.yml --name former
conda activate former
```

