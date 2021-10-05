

## Setup

Each example has a separate requirements file to avoid unnecessary extra installs for the smaller problems (MNIST,NLP).

The directories `mnist` `nlp` and `dlrm` contain MAP implementations (no sampling for variational inference) of our method and instructions to run the examples. 

The directory `tensor_layers` has the python package for the tensorized layers.

This implementation differs slightly from the results described in [our paper on Bayesian tensor rank determination for neural networks](https://arxiv.org/abs/2010.08689) in which many posterior samples are drawn. However the results are very similar.


## Issues?

Feel free to contact `colepshawkins@gmail.com` with any questions, or raise an issue.
