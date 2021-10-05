#%%
import bayesian_tensor_layers
import os
import bayesian_tensor_layers.low_rank_tensors
import bayesian_tensor_layers.layers
from bayesian_tensor_layers.layers import CPEmbedding
from bayesian_tensor_layers.low_rank_tensors import CP

#%%

shape = [[20, 20, 20], [10, 10]]

prior_type = 'log_uniform'

max_rank = 10

test = CP([10, 10, 10], max_rank=max_rank, prior_type=prior_type)

# %%

emb = CPEmbedding(shape=shape)

import torch
torch.std(emb.tensor.get_full())
emb.tensor.target_stddev
