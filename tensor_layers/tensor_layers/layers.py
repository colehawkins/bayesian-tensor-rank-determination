import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from . import low_rank_tensors
from .emb_utils import get_cum_prod,tensorized_lookup

class TensorizedLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    def forward(self, input, rank_update=True):

        if self.training and rank_update:
            self.tensor.update_rank_parameters()

        return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]),self.bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class TensorizedEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TensorizedEmbedding,self).__init__()

        self.shape = shape
        self.tensor_type=tensor_type

        target_stddev = np.sqrt(2/(np.prod(self.shape[0])+np.prod(self.shape[1])))

        if self.tensor_type=='TensorTrainMatrix':
            tensor_shape = shape
        else:
            tensor_shape = self.shape[0]+self.shape[1]

        self.tensor = getattr(low_rank_tensors,self.tensor_type)(tensor_shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        self.parameters = self.tensor.parameters

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = np.prod(self.shape[0])
        self.emb_quant = np.prod(self.shape[1])

        self.padding_idx = padding_idx
        self.naive = naive

        self.cum_prod = get_cum_prod(shape)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()


    def forward(self, x,rank_update=True):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        #x_ind = self.ind2sub(x)

#        full = self.tensor.get_full()
#        full = torch.reshape(full,[self.voc_quant,self.emb_quant])
#        rows = full[x]
        if hasattr(self.tensor,"masks"):
            rows = tensorized_lookup(x,self.tensor.get_masked_factors(),self.cum_prod,self.shape,self.tensor_type)
        else:
            rows = tensorized_lookup(x,self.tensor.factors,self.cum_prod,self.shape,self.tensor_type)
#        rows = gather_rows(self.tensor, x_ind)

        rows = rows.view(x.shape[0], -1)

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        

        return rows.to(x.device)
