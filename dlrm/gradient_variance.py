#%%
import tensorly as tl
import tensorly.contrib
import tensorly.decomposition
import torch_bayesian_tensor_layers
import numpy as np
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch_bayesian_tensor_layers.low_rank_tensors
import torch
import torch_bayesian_tensor_layers.layers

from torch_bayesian_tensor_layers.layers import TensorizedEmbedding

from train_then_compress_utils import tensor_decompose_and_replace_embedding
import time

torch.manual_seed(0)

t = time.time()

shape = [[100,100], [10,10]]
prior_type = 'log_uniform'
tensor_type = 'CP'
max_rank = 3
eta = 1e-2
inner_iter = 5
delta_approx=True
lr = 5e-3

if tensor_type != 'TensorTrainMatrix':
    shape[1] = [np.prod(shape[1])]



true_emb = TensorizedEmbedding(shape=shape,
                          tensor_type=tensor_type,
                          max_rank=1,
                          prior_type=prior_type,eta=eta)
trained_emb = TensorizedEmbedding(shape=shape,
                          tensor_type=tensor_type,
                          max_rank=max_rank,
                          prior_type=prior_type,eta=eta)

dims = shape[0]+shape[1]

if tensor_type=='TensorTrain':
    max_ranks = [1,max_rank,max_rank,max_rank,1]
elif tensor_type=='Tucker':
    max_ranks = len(dims)*[max_rank] 
elif tensor_type=='TensorTrainMatrix':
    max_ranks = [1,max_rank,max_rank,1]
elif tensor_type=='CP':
    max_ranks = max_rank

batch_size = np.prod(shape[0])//5
def get_batch_idx():
    n = np.prod(shape[0])
#    x_list = [x for x in range(n)]
    x_list = [random.randint(0, n - 1) for _ in range(batch_size)]
    input_values = torch.tensor(x_list)
    return input_values

true_emb.to('cuda')
trained_emb.to('cuda')

import torch.distributions as td

rank_parameter_std = torch.nn.Parameter(torch.ones(trained_emb.tensor.rank_parameter.shape))

rank_parameter_std.to('cuda')

rank_dist = td.HalfNormal(rank_parameter_std)

rank_prior_dist = td.HalfNormal(eta*torch.ones(rank_parameter_std.shape))

if delta_approx:
    opt = torch.optim.Adam(lr=lr,params = list(trained_emb.parameters()))
else:
    opt = torch.optim.Adam(lr=lr,params = list(trained_emb.parameters())[1:]+[rank_parameter_std])

grads_list = []
parameter_means = []


for ii in range(1000):

    grads_list.append([])

    if delta_approx:
        parameter_means.append(torch.square(trained_emb.tensor.rank_parameter).detach().cpu().numpy())
    else:
        parameter_means.append(torch.square(rank_dist.mean).detach().cpu().numpy())

    for jj in range(inner_iter):

        opt.zero_grad()
        
        input_values = get_batch_idx().to(true_emb.tensor.factors[0].device)

        if delta_approx:

            true_rows = true_emb.forward(input_values.view(-1, 1))
        else:

            true_rows = true_emb.forward(input_values.view(-1, 1),rank_update=False)


        learned_rows = trained_emb.forward(input_values.view(-1, 1))


        loss = torch.sum(torch.square(learned_rows-true_rows))

        kl_mult = 1e-2*torch.clamp(torch.tensor(ii/500),torch.tensor(0.0),torch.tensor(1.0))

        if delta_approx:

            loss+=kl_mult*(trained_emb.tensor.get_kl_divergence_to_prior())
        else:

            loss+=kl_mult*(trained_emb.tensor.get_kl_divergence_to_prior(rank_parameter=rank_dist.rsample().to('cuda'))+torch.sum(td.kl_divergence(rank_dist,rank_prior_dist)))

        loss.backward()

        grads_list[-1].append(trained_emb.tensor.factors[0].grad[0].detach().cpu().numpy())

    opt.step()

    if ii%100==0:
        print("%s of %s" % (ii,1000))
        print("Loss ",loss)
        print('RMSE ',torch.norm(learned_rows-true_rows)/torch.norm(true_rows))

        if delta_approx:
            print(trained_emb.tensor.estimate_rank(threshold=1e-6))
        else:
            print(rank_parameter_std)


variances = []

for grads in grads_list:
    var_mean = np.var(np.stack(grads),axis=0)
    variances.append(var_mean)
    #variances.append(np.var(np.stack(grads),axis=0))

stacked_parameter_means = np.concatenate(parameter_means,axis=0).T

stacked_variances = np.stack(variances,axis=1)


import matplotlib.pyplot as plt

for ii in range(3):
    plt.figure()
    plt.title('Variance %s' % (ii))
    plt.semilogy(stacked_variances[ii])
    plt.figure()
    plt.title('Parameter %s' % (ii))
    plt.semilogy(stacked_parameter_means[ii])

if delta_approx:
    delta_variance = stacked_variances
else:
    sampling_variance = stacked_variances

for ii in range(3):
    plt.figure()
    plt.title('Variance rank %s' % (ii))
    plt.semilogy(delta_variance[ii],label='Delta variance')
    plt.semilogy(sampling_variance[ii],label='Sampling variance')
    plt.legend()
#%%
trained_emb.tensor.rank_parameter

trained_emb.tensor.factors[0].grad[0].detach().cpu().numpy()

"""

factors = trained_emb.tensor.factors

rank_variables = trained_emb.tensor.rank_parameters

threshold = 1e-5

masks = [
            torch.tensor(torch.square(x)>threshold, dtype=torch.float32)
            for x in rank_variables
]
        

for mask in masks:
    print(mask.shape)
    print(mask)


factors = [x*y for x,y in zip(factors,masks)]+[masks[-1].view([-1,1,1,1])*factors[-1]]

"""

tmp = trained_emb.tensor.get_full()

print("Pruning norm difference ",torch.norm(tmp-full)/torch.norm(full))


opt = torch.optim.Adam(lr=lr*1e-2,params = trained_emb.parameters())


for ii in range(1000):

    opt.zero_grad()

    input_values = get_batch_idx().to(true_emb.tensor.factors[0].device)

    learned_rows = trained_emb.forward(input_values.view(-1, 1))

    true_rows = true_emb.forward(input_values.view(-1, 1))

    loss = torch.sum(torch.square(learned_rows-true_rows))

    loss.backward()

    opt.step()

    if ii%100==0:
        print(torch.sum(torch.square(learned_rows-true_rows)))
        print("Fine tune loss ",trained_emb.tensor.estimate_rank())


# %%

"""
tensorized_embedding = tensor_decompose_and_replace_embedding(true_emb,tensor_type,shape,max_ranks)


full_1 = true_emb.weight#tensorized_embedding.tensor.get_full()


full_2 = torch.reshape(tensorized_embedding.tensor.get_full(),[np.prod(shape[0]),np.prod(shape[1])])

print(torch.norm(full_1-full_2[:full_1.shape[0]])/torch.norm(full_1))

print(time.time()-t)

padded_tensor = pad_tensor(init_tensor,shape)
reshape_dims = get_reshape_dims(tensor_type,shape)
reshaped_padded_tensor = torch.reshape(padded_tensor,reshape_dims)

if tensor_type!='TensorTrainMatrix':
    factors = tensor_decompose(reshaped_padded_tensor,tensor_type,max_ranks)
elif tensor_type=='TensorTrainMatrix':
#    dims = [x*y for x,y in zip(shape[0],shape[1])]
    factors = tensor_decompose(reshaped_padded_tensor,tensor_type,max_ranks)
#    factors = [torch.reshape(factor,[max_ranks[i],shape[0][i],shape[1][i],max_ranks[i+1]]) for i,factor in enumerate(factors)]
else: 
    raise NotImplementedError

if tensor_type!='Tucker':
    for x in factors:
        print(x.shape)

else:
    print(factors[0].shape)
    for x in factors[1]:
        print(x.shape)


if tensor_type=='TensorTrain':
    full = tl.tt_to_tensor(factors)

elif tensor_type=='Tucker':
    full = tl.tucker_to_tensor(factors)

elif tensor_type=='CP':
    full = tl.kruskal_to_tensor((None,factors))

elif tensor_type=='TensorTrainMatrix':
    full = tl.tt_matrix.tt_matrix_to_tensor(factors)


full = torch.reshape(full,[np.prod(shape[0]),np.prod(shape[1])])

if tensor_type!='TensorTrainMatrix':
    factors = tensor_decompose(reshaped_padded_tensor,tensor_type,max_ranks,dims)
elif tensor_type:
    dims = [x*y for x,y in zip(shape)]
else: 
    raise NotImplementedError

factors = tensor_decompose(padded_reshaped_tensor,tensor_type,max_ranks,shape)

for x in factors:
    print(x.shape)

"""