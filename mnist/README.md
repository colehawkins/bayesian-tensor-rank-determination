The main file is modified from a standard pytorch [MNIST example](https://github.com/pytorch/examples/tree/master/mnist)


## Setup 

Install the requirements and set up a conda environment:
```
conda env update -f requirements.yml
```

Install the tensor layers package. You will need to move up one directory, then return to this directory to run the MNIST example.
```
cd ../tensor_layers
pip install -e .
```

This setup has been tested on a clean g4dn-xlarge AWS instance running Ubuntu 18.04 (deep learning AMI).

## Run the code

The appropriate scripts are all in the scripts directory. They call the `train.py` file with the appropriate command line arguments. You should run the scripts from this directory (`mnist`)

For example, to run the CP format:
```
bash scripts/cp.sh
```
The TT,TTM, and Tucker formats are similar.

On the first run the MNIST dataset should download automatically
