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

You can clone the code and run the experiments from the root directory. E.g. 

```
python experiments/classify.py
```

Hyperparameters are passed as command line arguments. The defaults should work well. The classification data is 
automatically downloaded, the wikipedia data is included in the repository.

You should be able to install as a package as well, with 
```
pip install git+https://github.com/pbloem/former
```
but I haven't tried this. It's probably easier to just copy over the code you need. Let me know if you need this for anything and it doesn't work. 

## Requirements

The following should install all requirements 
```pip install torch tb-nightly tqdm numpy torchtext```

Let me know if I missed anything. 