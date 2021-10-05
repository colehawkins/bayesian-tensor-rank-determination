import numpy as np
import torch
import tensorly as tl
import tensorly.decomposition
import torch_bayesian_tensor_layers

from torch_bayesian_tensor_layers.layers import TensorizedEmbedding

def get_reshape_dims(tensor_type,shape):
    if tensor_type!='TensorTrainMatrix':
        reshape_dims = shape[0]+shape[1]
    elif tensor_type=='TensorTrainMatrix':
        reshape_dims = shape[0]+shape[1]#[x*y for x,y in zip(shape[0],shape[1])]
    else: 
        raise NotImplementedError

    return reshape_dims

def pad_tensor(init_tensor,shape):

    print('Shape ',shape)
    print("init tensor shape",init_tensor.shape)

    padding_size = np.prod(shape[0])-init_tensor.shape[0]

    if padding_size>0:
        padding_tensor = init_tensor[-padding_size:]
        padded_tensor = torch.cat([init_tensor,padding_tensor],axis=0)
    else:
        padded_tensor = init_tensor

    return padded_tensor

@torch.no_grad()
def tensor_decompose(to_decompose,tensor_type,max_ranks):

    if tensor_type=='TensorTrain':
        factors = tl.decomposition.tensor_train(to_decompose,max_ranks)
    elif tensor_type=='Tucker':
        factors = tl.decomposition.tucker(to_decompose,max_ranks)
    elif tensor_type=='TensorTrainMatrix':
        factors = tl.decomposition.tensor_train_matrix(to_decompose,max_ranks)
    elif tensor_type=='CP':
        _,factors = tl.decomposition.parafac(to_decompose,max_ranks,n_iter_max=10,init='random')

    return factors

@torch.no_grad()
def tensor_decompose_and_replace_embedding(embedding,tensor_type,shape,max_ranks):

    init_tensor = embedding.weight.detach()
    padded_tensor = pad_tensor(init_tensor,shape)
    reshape_dims = get_reshape_dims(tensor_type,shape)
    reshaped_padded_tensor = torch.reshape(padded_tensor,reshape_dims)

    factors = tensor_decompose(reshaped_padded_tensor,tensor_type,max_ranks)

    tensorized_embedding = TensorizedEmbedding(init='nn',shape=shape,tensor_type=tensor_type,max_rank=max_ranks)

    with torch.no_grad():

        if tensor_type=='Tucker':

            tensorized_embedding.tensor.factors[0].copy_(factors[0].data)
            
            for x,y in zip(tensorized_embedding.tensor.factors[1],factors[1]):
                x.copy_(y.data)
        else:
            for x,y in zip(tensorized_embedding.tensor.factors,factors):
                print(x.shape,y.shape)
                x.copy_(y.data)

    return tensorized_embedding
