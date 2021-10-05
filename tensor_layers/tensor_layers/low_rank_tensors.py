#%%
import torch 
import tensorly as tl
tl.set_backend('pytorch')
import tensorly.random, tensorly.decomposition
from abc import abstractmethod, ABC
import torch.distributions as td
Parameter = torch.nn.Parameter
import numpy as np
from .truncated_normal import TruncatedNormal

class LowRankTensor(torch.nn.Module):
    def __init__(self,
                 dims,
                 prior_type=None,
                 init_method='random',
                 learned_scale=True,
                 **kwargs):

        super(LowRankTensor, self).__init__()

        self.eps = 1e-12
        self.dims = dims
        self.order = len(self.dims)
        self.prior_type = prior_type

        for x in kwargs:
            setattr(self, x, kwargs.get(x))

        self.trainable_variables = []

        self._build_factors()
        self._build_factor_distributions()
        self._build_low_rank_prior()

        self.trainable_variables = torch.nn.ParameterList(self.trainable_variables)

    def get_relative_mse(self, sample_tensor):
        return torch.norm(self.get_full() -
                              sample_tensor) / torch.norm(sample_tensor)

    def add_variable(self, initial_value,trainable=True):

        #add weight using torch interface
        new_variable = Parameter(initial_value.clone().detach(),requires_grad=trainable)

        self.trainable_variables.append(new_variable)

        return new_variable

    @abstractmethod
    def _build_factors(self):
        pass

    @abstractmethod
    def _build_factor_distributions(self):
        pass

    @abstractmethod
    def _build_low_rank_prior(self):
        pass

    @abstractmethod
    def get_full(self):
        pass

    @abstractmethod
    def sample_full(self):
        pass

    @abstractmethod
    def get_rank(self, threshold=1e-5):
        pass

    @abstractmethod
    def prune_ranks(self, threshold=1e-5):
        pass

    @abstractmethod
    def get_kl_divergence_to_prior(self):
        pass

    @abstractmethod
    def get_parameter_savings(self):
        pass


class CP(LowRankTensor):
    def __init__(self, dims, max_rank, learned_scale=True,**kwargs):

        self.max_rank = max_rank
        self.learned_scale=learned_scale
        self.max_ranks = max_rank
        super().__init__(dims, **kwargs)
        self.tensor_type = 'CP'
        

    #@tf.function
    def get_full(self):
        return tl.kruskal_to_tensor((self.weights, self.factors))

    def get_masked_factors(self):
        return [self.weights*factor for factor in self.factors]


    def get_rank_variance(self):
        return torch.square(torch.relu(self.rank_parameter))


    def estimate_rank(self, threshold=1e-5):

        return int(sum(sum(self.get_rank_variance() > threshold)))

    def prune_ranks(self, threshold=1e-5):

        mask = (self.get_rank_variance()>threshold).float()

        mask.to(self.rank_parameter.device)

        self.masks = [mask] 

        self.weights = torch.squeeze(mask)
        
    def get_parameter_savings(self, threshold=1e-5):
        
        rank_difference = self.max_rank-self.estimate_rank()
        savings_per_dim = [rank_difference*x for x in self.dims]
        low_rank_savings = sum(savings_per_dim)

        tensorized_savings = np.prod(self.dims)-sum([self.max_rank*x for x in self.dims])

        return low_rank_savings,tensorized_savings

    def _random_init(self):

        random_init = tl.random.random_kruskal(self.dims,
                                               self.max_rank,
                                               random_state=getattr(
                                                   self, 'seed', None))

        if hasattr(self, 'target_norm'):
            curr_norm = torch.norm(
                tl.kruskal_to_tensor(
                    (torch.ones([self.max_rank]), random_init[1])))
            mult_factor = torch.pow(
                getattr(self, 'target_norm') / curr_norm, 1 / len(self.dims))
            scaled_factors = [mult_factor * x for x in random_init[1]]
            random_init = (torch.ones([self.max_rank]), scaled_factors)

        return random_init

    def _nn_init(self):


        if hasattr(self,"target_stddev"):
            pass
        else:
            self.target_stddev = 0.05
    
        factor_stddev = torch.pow(
            self.target_stddev / torch.sqrt(torch.tensor(1.0 * self.max_rank)),
            1.0 / len(self.dims))
        self.factor_stddev = factor_stddev
        
        initializer_dist = TruncatedNormal(loc=0.0,scale=factor_stddev,a=-3.0*factor_stddev,b=3.0*factor_stddev)
        
        init_factors = (torch.ones([self.max_rank]), [
            initializer_dist.sample([x, self.max_rank]) for x in self.dims
        ])

        return init_factors

    def _build_factors(self):

        if hasattr(self, 'initialization_method'):
            if self.initialization_method == 'random':
                self.weights, self.factors = self._random_init()
            elif self.initialization_method == 'parafac':
                self.initialization_tensor = torch.reshape(self.initialization_tensor,self.dims)
                self.weights, self.factors = tl.decomposition.parafac(
                    self.initialization_tensor, self.max_rank, init='random')
            elif self.initialization_method == 'parafac_svd':
                self.initialization_tensor = torch.reshape(self.initialization_tensor,self.dims)
                self.weights, self.factors = tl.decomposition.parafac(
                    self.initialization_tensor, self.max_rank, init='svd')
            elif self.initialization_method == 'nn':
                self.weights, self.factors = self._nn_init()
            else:
                raise (ValueError("Initialization method not supported."))
        else:
            self.weights, self.factors = self._random_init()

        #convert all to tensorflow variable
        self.factors = [self.add_variable(x) for x in self.factors]
        self.weights = None


    def _build_factor_distributions(self):

        factor_scale_multiplier = 1e-9

        factor_scales = [
            self.add_variable(factor_scale_multiplier *
                              torch.ones(factor.shape),trainable=self.learned_scale) for factor in self.factors
        ]

        self.factor_distributions = []

        for factor, factor_scale in zip(self.factors, factor_scales):
            self.factor_distributions.append(
                td.Independent(td.Normal(
                    loc=factor,
                    scale=factor_scale),
                                reinterpreted_batch_ndims=2))


    def _build_low_rank_prior(self):

        self.rank_parameter = self.add_variable(torch.sqrt(self.get_rank_parameters_update().clone().detach()).view([1,self.max_rank]),trainable=False)# Parameter(torch.sqrt(torch.tensor(self.get_rank_parameters_update())).view([1,self.max_rank]))

        self.factor_prior_distributions = []

        for x in self.dims:
            zero_mean = torch.zeros([x, self.max_rank])
            base_dist = td.Normal(loc=zero_mean,scale=self.rank_parameter)
            independent_dist = td.Independent(base_dist,reinterpreted_batch_ndims=2)
            self.factor_prior_distributions.append(independent_dist)#td.Independent(base_dist,reinterpreted_batch_ndims=2))

    def sample_full(self):
        return tl.kruskal_to_tensor(
            (self.weights, [x.rsample() for x in self.factor_distributions]))

    def get_rank_parameters_update(self):
        def half_cauchy():


            M = torch.sum(torch.stack([torch.sum(torch.square(x.mean) + torch.square(x.stddev),
                              dim=0) for x in self.factor_distributions]),dim=0)

            D = 1.0 * sum(self.dims)

            update = (M - D * self.eta**2 + torch.sqrt(torch.square(M) + (2.0 * D + 8.0) * torch.square(torch.tensor(self.eta)) * M +torch.pow(torch.tensor(self.eta), 4.0) * torch.square(torch.tensor(D)))) / (2.0 * D + 4.0)

            return update

        def log_uniform():

            M = torch.sum(torch.stack([torch.sum(torch.square(x.mean) + torch.square(x.stddev),
                              dim=0) for x in self.factor_distributions]),dim=0)

            D = 1.0 * (sum(self.dims) + 1.0)

            update = M / D

            return update

        if self.prior_type == 'log_uniform':
            return log_uniform()
        elif self.prior_type == 'half_cauchy':
            return half_cauchy()
        else:
            raise ValueError("Prior type not supported")

    def update_rank_parameters(self):


        with torch.no_grad():

            rank_update = self.get_rank_parameters_update()
            sqrt_parameter_update = torch.sqrt((1 - self.em_stepsize) * self.rank_parameter.data**2 + self.em_stepsize * rank_update)
            self.rank_parameter.data.sub_(self.rank_parameter.data)
            self.rank_parameter.data.add_(sqrt_parameter_update.to(self.rank_parameter.device))

    def get_rank(self, threshold=1e-4):
        return len(torch.where(self.get_rank_variance() > threshold))

    def get_kl_divergence_to_prior(self,rank_parameter=None):

        if rank_parameter is None:
            rank_parameter = self.rank_parameter 
        else:
            pass

        kl_sum= 0.0

        for p in self.factor_distributions:
            var_ratio = (p.stddev / rank_parameter).pow(2)
            t1 = ((p.mean ) / rank_parameter).pow(2)
            kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))
            kl_sum+=kl

        return kl_sum


class TensorTrain(LowRankTensor):
    def __init__(self, dims, max_rank,learned_scale=True, **kwargs):

        self.learned_scale=learned_scale
        self.max_rank = max_rank

        if type(self.max_rank)==int:
            self.max_ranks = [1]+(len(dims)-1)*[self.max_rank]+[1]
        else:
            assert(type(max_rank)==list)
            self.max_ranks = max_rank
            self.max_rank = max(self.max_rank)


        super().__init__(dims, **kwargs)
        self.tensor_type = 'TensorTrain'

    def get_full(self):

        if hasattr(self,"masks"):
            factors = self.get_masked_factors()             
            return tl.tt_to_tensor(factors)
        else:
            return tl.tt_to_tensor(self.factors)

    def get_masked_factors(self):
        factors = [x*y for x,y in zip(self.factors,self.masks)]+[self.masks[-1].view([-1,1,1])*self.factors[-1]]
        return factors

    def estimate_rank(self, threshold=1e-4):

        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]


    def get_parameter_savings(self,threshold=1e-4):
        
        rank_estimates = [1]+self.estimate_rank(threshold=threshold)+[1]

        reduced_rank_parameters = 0
        total_tt_parameters = sum([np.prod(x.shape) for x in self.factors])

        for i,x in enumerate(self.dims):
            reduced_rank_parameters+= rank_estimates[i]*x*rank_estimates[i+1]
            


        return total_tt_parameters-reduced_rank_parameters,np.prod(self.dims)-total_tt_parameters


    def _nn_init(self):

        if hasattr(self,"target_stddev"):
            pass
        else:
            self.target_stddev = 0.05

        self.target_stddev = torch.tensor(self.target_stddev)

        factor_stddev = torch.pow(
            torch.pow(torch.tensor(1.0 * self.max_rank),torch.tensor( -self.order + 1)) *
            torch.square(self.target_stddev), 1 / (2.0 * self.order))
        self.factor_stddev = factor_stddev

        sizes = [[1, self.dims[0], self.max_ranks[1]]] + [[self.max_ranks[i+1], x, self.max_ranks[i+2]] for i,x in enumerate(self.dims[1:-1])] + [[self.max_ranks[-2], self.dims[-1], 1]]

        initializer_dist = TruncatedNormal(loc=0.0,
                                               scale=factor_stddev,
                                               a=-self.order * factor_stddev,
                                               b=self.order * factor_stddev)
        init_factors = [initializer_dist.sample(x) for x in sizes]

        return init_factors

    def _random_init(self):

        factors = tl.random.random_mps(self.dims,
                                       self.max_ranks,
                                       full=False,
                                       random_state=getattr(
                                           self, 'seed', None))

        if hasattr(self, 'target_norm'):
            curr_norm = torch.norm(tl.mps_to_tensor(factors))
            multiplier = torch.pow(self.target_norm / curr_norm, 1 / self.order)
            factors = [multiplier * x for x in factors]

        return factors

    def _build_factors(self):

        if hasattr(self, 'initialization_method'):
            if self.initialization_method == 'random':
                self.factors = self._random_init()
            elif self.initialization_method == 'nn':
                self.factors = self._nn_init()
            elif self.initialization_method == 'svd':
                self.initialization_tensor = torch.reshape(self.initialization_tensor,self.dims)
                self.factors = tensorly.decomposition.matrix_product_state(
                    self.initialization_tensor, self.max_ranks)
            else:
                raise (ValueError("Initialization method not supported."))
        else:
            self.factors = self._random_init()

        self.factors = [self.add_variable(x) for x in self.factors]

        self.sizes = [x.shape for x in self.factors]

    def _build_factor_distributions(self):

        factor_scale_init = 1e-9

        factor_scales = [
            self.add_variable(factor_scale_init * torch.ones(factor.shape),trainable=self.learned_scale)
            for factor in self.factors
        ]

        self.factor_distributions = []

        for factor, factor_scale in zip(self.factors, factor_scales):
            self.factor_distributions.append(
                td.Independent(td.Normal(
                    loc=factor,
                    scale=factor_scale),
                                reinterpreted_batch_ndims=3))

    def _build_low_rank_prior(self):

        self.rank_parameters = [
            self.add_variable(torch.sqrt(x.clone().detach()),trainable=False)
            for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = []

        for i in range(len(self.dims) - 1):

            self.factor_prior_distributions.append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[i].shape),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=3))

        self.factor_prior_distributions.append(
            td.Independent(td.Normal(
                loc=torch.zeros(self.factors[-1].shape),
                scale=self.rank_parameters[-1].unsqueeze(1).unsqueeze(2)),
                            reinterpreted_batch_ndims=3))

    def sample_full(self):
        
        factors = [x.rsample() for x in self.factor_distributions]

        if hasattr(self,"masks"):
            factors = [x*y for x,y in zip(factors,self.masks)]+[self.masks[-1].view([-1,1,1])*factors[-1]]
        
        return tl.tt_to_tensor(factors)
        

    def get_rank_parameters_update(self):

        updates = []
        
        for i in range(len(self.dims) - 1):

            M = torch.sum(torch.square(self.factor_distributions[i].mean) +
                              torch.square(self.factor_distributions[i].stddev),
                              axis=[0, 1])

            if i == len(self.dims) - 2:
                D = self.max_ranks[i] * self.dims[i] + self.dims[i + 1]
                M += torch.sum(
                    torch.square(self.factor_distributions[i + 1].mean) +
                    torch.square(self.factor_distributions[i + 1].stddev),
                    axis=[1, 2])
            else:
                D = self.max_ranks[i] * self.dims[i]

            if self.prior_type == 'log_uniform':
                update = M / (D + 1)

            elif self.prior_type == 'half_cauchy':
                update = (M - (self.eta**2) * D +
                          torch.sqrt(M**2 + (M * self.eta**2) * (2.0 * D + 8.0) +
                                  (D**2.0) * (self.eta**4.0))) / (2 * D + 4.0)

            updates.append(update)

        return updates

    def update_rank_parameters(self):

        with torch.no_grad():
            rank_updates = self.get_rank_parameters_update()

            for rank_parameter, rank_update in zip(self.rank_parameters, rank_updates):


                sqrt_parameter_update = torch.sqrt((1 - self.em_stepsize) * rank_parameter.data**2 + self.em_stepsize * rank_update)
                rank_parameter.data.sub_(rank_parameter.data)
                rank_parameter.data.add_(sqrt_parameter_update.to(rank_parameter.device))


    def get_rank(self, threshold=1e-5):
        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]

    def prune_ranks(self, threshold=1e-5):
        self.masks =[(torch.square(x)>threshold).detach().clone().float().to(x.device) for x in self.rank_parameters]

    def get_kl_divergence_to_prior(self):

        kl_sum= 0.0

        appended_rank_parameters = self.rank_parameters+[self.rank_parameters[-1].unsqueeze(1).unsqueeze(2)]

        for p,rank_parameter in zip(self.factor_distributions,appended_rank_parameters):
            var_ratio = (p.stddev / rank_parameter).pow(2)
            t1 = ((p.mean ) / rank_parameter).pow(2)
            kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))
            kl_sum+=kl

        

        return kl_sum



class Tucker(LowRankTensor):
    def __init__(self, dims, max_rank,learned_scale=True, **kwargs):

        self.max_rank = max_rank
        self.learned_scale=learned_scale

        if type(self.max_rank)==int:
            self.max_ranks = len(dims)*[self.max_rank]
        else:
            assert(type(max_rank)==list)
            self.max_ranks = max_rank
            self.max_rank = max(self.max_rank)

        super().__init__(dims, **kwargs)
        self.tensor_type = 'tucker'

    def get_full(self):


        if hasattr(self,"masks"):
            factors = self.get_masked_factors()
        else:
            factors = list(self.factors)
        
        return tl.tucker_to_tensor(factors)

    def get_masked_factors(self):

        factors = list(self.factors)
        factors[1] = [x*factor for x,factor in zip(self.masks,self.factors[1])]
        return factors


    def estimate_rank(self, threshold=1e-4):

        return [int(sum(torch.square(x.detach().clone()) > threshold)) for x in self.rank_parameters]

    def get_parameter_savings(self,threshold=1e-4):
        
        rank_differences = [self.max_ranks[i]-x for i,x in enumerate(self.estimate_rank(threshold=threshold))]
        factor_savings = sum([self.dims[i]*x for i,x in enumerate(rank_differences)])
        
        core_parameters = np.prod(self.max_ranks)#**(len(self.dims)) 
        core_savings = core_parameters-np.prod(self.estimate_rank(threshold=threshold))
        tensorized_savings = np.prod(self.dims)-core_parameters-sum([self.max_ranks[i]*x for i,x in enumerate(self.dims)])


        return factor_savings+core_savings,tensorized_savings


    def prune_ranks(self, threshold=1e-4):
        self.masks =[(torch.square(x)>threshold).detach().clone().float().to(x.device) for x in self.rank_parameters]

    def _nn_init(self):

        if hasattr(self,"target_stddev"):
            pass
        else:
            self.target_stddev = 0.05

        self.target_stddev = torch.tensor(self.target_stddev)

        factor_stddev = torch.pow(
            torch.pow(torch.tensor(1.0 * self.max_rank),torch.tensor( -self.order)) *
            torch.square(self.target_stddev), torch.tensor(1 / (2.0 * (self.order + 1))))
        initializer_dist = TruncatedNormal(loc=0.0,
                                               scale=factor_stddev,
                                               a=-self.order * factor_stddev,
                                               b=self.order * factor_stddev)
        sizes = (self.max_ranks, [[dim, rank]
                                                for dim,rank in zip(self.dims,self.max_ranks)])
        init_factors = (initializer_dist.sample(sizes[0]),
                        [initializer_dist.sample(x) for x in sizes[1]])

        return init_factors

    def _random_init(self):

        random_init = tl.random.random_tucker(self.dims,
                                              self.max_ranks,
                                              full=False,
                                              random_state=getattr(
                                                  self, 'seed', None))

        if hasattr(self, 'target_norm'):
            curr_norm = torch.norm(tl.tucker_to_tensor(random_init))
            multiplier = torch.pow(self.target_norm / curr_norm, 1 / self.order)

            random_init = (random_init[0],
                           [multiplier * x for x in random_init[1]])

        return random_init

    def _build_factors(self):
        if hasattr(self, 'initialization_method'):
            if self.initialization_method == 'random':
                self.factors = self._random_init()
            elif self.initialization_method == 'nn':
                self.factors = self._nn_init()
            elif self.initialization_method == 'hooi':
                self.initialization_tensor = torch.reshape(self.initialization_tensor,self.dims)
                self.factors = tl.decomposition.tucker(
                    self.initialization_tensor, self.max_ranks)
        else:
            self.factors = self._random_init()

        self.factors = (self.add_variable(self.factors[0]),
                        [self.add_variable(x) for x in self.factors[1]])

    def _build_factor_distributions(self):

        factor_scale_init = 1e-9

        factor_scales = (self.add_variable(
            factor_scale_init * torch.ones(self.factors[0].shape),trainable=self.learned_scale), [
                self.add_variable(factor_scale_init * torch.ones(factor.shape),trainable=self.learned_scale)
                for factor in self.factors[1]
            ])

        self.factor_distributions = (td.Independent(
            td.Normal(loc=self.factors[0],
                       scale=factor_scales[0]),
            reinterpreted_batch_ndims=len(self.dims)), [])

        for factor, factor_scale in zip(self.factors[1], factor_scales[1]):
            self.factor_distributions[1].append(
                td.Independent(td.Normal(
                    loc=factor,
                    scale=factor_scale),
                                reinterpreted_batch_ndims=2))

    def _build_low_rank_prior(self, core_prior=10.0):

        self.rank_parameters = [
            self.add_variable(torch.sqrt(x.clone().detach()),trainable=False) for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = (td.Independent(
            td.Normal(loc=torch.zeros(self.factors[0].shape), scale=core_prior),
            reinterpreted_batch_ndims=len(self.dims)), [])

        for i in range(len(self.dims)):

            self.factor_prior_distributions[1].append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[1][i].shape),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=2))

    def sample_full(self):

        sample_factors = [self.factor_distributions[0].rsample(),[x.rsample() for x in self.factor_distributions[1]]]
        
        if hasattr(self,"masks"):
            raise NotImplementedError
            sample_factors[0] = tf.multiply(tl.kruskal_to_tensor(([1.0],[tf.expand_dims(x,axis=1) for x in self.masks])),sample_factors[0])

        return tl.tucker_to_tensor(sample_factors)

    def get_rank_parameters_update(self):

        updates = []

        for i in range(len(self.dims)):

            M = torch.sum(
                torch.square(self.factor_distributions[1][i].mean) +
                torch.square(self.factor_distributions[1][i].stddev),
                axis=0)

            if self.prior_type == 'log_uniform':
                update = M / (self.dims[i] + 1)

            elif self.prior_type == 'half_cauchy':
                update = (M - (self.eta**2) * self.dims[i] +
                          torch.sqrt(M**2 + (M * self.eta**2) *
                                  (2.0 * self.dims[i] + 8.0) +
                                  (self.dims[i]**2.0) *
                                  (self.eta**4.0))) / (2 * self.dims[i] + 4.0)

            updates.append(update)

        return updates


    def update_rank_parameters(self):

        with torch.no_grad():

            rank_updates = self.get_rank_parameters_update()
            for rank_parameter, rank_update in zip(self.rank_parameters, rank_updates):
                
                sqrt_parameter_update = torch.sqrt((1 - self.em_stepsize) * rank_parameter.data**2 + self.em_stepsize * rank_update)
                rank_parameter.data.sub_(rank_parameter.data)
                rank_parameter.data.add_(sqrt_parameter_update.to(rank_parameter.device))

    def get_rank(self, threshold=1e-4):
        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]

    def get_kl_divergence_to_prior(self):

        kl_sum = 0.0

        for p,rank_parameter in zip(self.factor_distributions[1],self.rank_parameters):
            var_ratio = (p.stddev / rank_parameter).pow(2)
            t1 = ((p.mean ) / rank_parameter).pow(2)
            kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))
            kl_sum+=kl

        return kl_sum



class TensorTrainMatrix(LowRankTensor):
    def __init__(self, dims, max_rank,learned_scale=True, **kwargs):

        self.max_rank = max_rank
        self.learned_scale=learned_scale

        if type(self.max_rank)==int:
            self.max_ranks = [1]+(len(dims[0])-1)*[self.max_rank]+[1]
        else:
            assert(type(max_rank)==list)
            self.max_ranks = max_rank
            self.max_rank = max(self.max_rank)

        self.dims1,self.dims2 = dims
        self.shape = [np.prod(self.dims1),np.prod(self.dims2)]
        assert(len(self.dims1)==len(self.dims2))

        super().__init__(dims, **kwargs)
        self.tensor_type = 'TensorTrainMatrix'
        self.order = len(self.dims1)

    def get_parameter_savings(self,threshold=1e-4):
        
        rank_estimates = [1]+self.estimate_rank(threshold=1e-4)+[1]

        reduced_rank_parameters = 0
        total_tt_parameters = sum([np.prod(x.shape) for x in self.factors])

        for i,x in enumerate(zip(self.dims1,self.dims2)):
            reduced_rank_parameters+= rank_estimates[i]*x[0]*x[1]*rank_estimates[i+1]
            
        return total_tt_parameters-reduced_rank_parameters,np.prod(self.dims)-total_tt_parameters

    def get_masked_factors(self):
        return [x*y for x,y in zip(self.factors,self.masks)]+[self.masks[-1].view([-1,1,1,1])*self.factors[-1]]

    def full_from_factors(self,factors):

        if hasattr(self,'masks'):
#            factors = [x*y for x,y in zip(factors,self.masks)]+[self.masks[-1].view([-1,1,1,1])*factors[-1]]
            factors = self.get_masked_factors()

        num_dims = len(self.dims1)

        ranks = [x[0] for x in self.rank_pairs]
        shape = self.shape
        raw_shape = self.dims

        res = factors[0]

        for i in range(1, num_dims):
            res = torch.reshape(res, (-1, ranks[i]))
            curr_core = torch.reshape(factors[i], (ranks[i], -1))
            res = torch.matmul(res, curr_core)


        intermediate_shape = []
        for i in range(num_dims):
            intermediate_shape.append(raw_shape[0][i])
            intermediate_shape.append(raw_shape[1][i])
        res = torch.reshape(res, intermediate_shape)
        transpose = []
        for i in range(0, 2 * num_dims, 2):
            transpose.append(i)
        for i in range(1, 2 * num_dims, 2):
            transpose.append(i)
        res = torch.Tensor.permute(res, transpose)
        res = torch.reshape(res, shape)

        return res

    def get_full(self):

        return self.full_from_factors(self.factors)

    def estimate_rank(self, threshold=1e-5):

        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]

    def prune_ranks(self, threshold=1e-5):

        self.masks = [
            (torch.square(x)>threshold).detach().clone().float().to(x.device)
            for x in self.rank_parameters
        ]

    def _nn_init(self):

        if hasattr(self,"target_stddev"):
            pass
        else:
            self.target_stddev = 0.05

        cr_exponent = -1.0 / (2 * len(self.dims1))
        var = np.prod(self.max_ranks) ** cr_exponent
        core_stddev = self.target_stddev ** (1.0 / len(self.dims1)) * var

        self.target_stddev = torch.tensor(self.target_stddev)
        """
        factor_stddev = torch.pow(
            torch.pow(torch.tensor(1.0 * self.max_rank),torch.tensor( -self.order + 1)) *
            torch.square(self.target_stddev), 1.0 / torch.tensor(2.0 * self.order))
        """
        self.factor_stddev = core_stddev

        sizes = [[1, self.dims1[0],self.dims2[0], self.max_ranks[1]]] + [
            [self.max_ranks[i+1], x,y, self.max_ranks[i+2]] for i,(x,y) in enumerate(zip(self.dims1[1:-1],self.dims2[1:-1]))
        ] + [[self.max_ranks[-2], self.dims1[-1],self.dims2[-1], 1]]


        initializer_dist = TruncatedNormal(loc=0.0,
                                               scale=self.factor_stddev,
                                               a=-4.0 * self.factor_stddev,
                                               b=4.0 * self.factor_stddev)
        
        init_factors = [initializer_dist.sample(x) for x in sizes]

        return init_factors

    def _glorot_init(self):
        raise NotImplementedError
        initializer = tf.keras.initializers.glorot_normal(0)

        init_factors = [tf.reshape(initializer([self.sizes[0][1],self.sizes[0][2],self.sizes[0][0],self.sizes[0][3]]),self.sizes[0])]
        
        for x in self.sizes[1:]:

            unshaped = initializer([np.prod(x[0:2]),np.prod(x[2:])])
            init_factors.append(tf.reshape(unshaped,x))

        return init_factors

    def _random_init(self):

        factors = [td.Normal(0.0,1.0).sample(sz) for sz in self.sizes]

        if hasattr(self, 'target_norm'):
            curr_norm = torch.norm(self.full_from_factors(factors))
            multiplier = torch.pow(self.target_norm / curr_norm, 1 / self.num_cores)
            factors = [multiplier * x for x in factors]

        return factors

    def _build_factors(self):

        self.num_cores = len(self.dims1)
        self.rank_pairs = list(zip(self.max_ranks,self.max_ranks[1:]))

        self.sizes = [(rank_pair[0],i,j,rank_pair[1]) for rank_pair,i,j in zip(self.rank_pairs,self.dims1,self.dims2)]

        if hasattr(self, 'initialization_method'):
            if self.initialization_method == 'random':
                self.factors = self._random_init()
            elif self.initialization_method == 'nn':
                self.factors = self._nn_init()
            elif self.initialization_method == 'glorot':
                self.factors = self._glorot_init()
            else:
                raise (ValueError("Initialization method not supported."))
        else:
            self.factors = self._random_init()

        #convert all to tensorflow variable
        

        self.factors = [self.add_variable(x) for x in self.factors]



    def _build_factor_distributions(self):

        factor_scale_init = 1e-9

        factor_scales = [
            self.add_variable(factor_scale_init * torch.ones(factor.shape),trainable=self.learned_scale)
            for factor in self.factors
        ]

        self.factor_distributions = []

        for factor, factor_scale in zip(self.factors, factor_scales):
            self.factor_distributions.append(
                td.Independent(td.Normal(
                    loc=factor,
                    scale=factor_scale),
                                reinterpreted_batch_ndims=4))

    def _build_low_rank_prior(self):

        self.rank_parameters = [
            self.add_variable(x.clone().detach(),trainable=False)
            for x in self.get_rank_parameters_update()
        ]

        self.factor_prior_distributions = []

        for i in range(self.num_cores - 1):

            self.factor_prior_distributions.append(
                td.Independent(td.Normal(
                    loc=torch.zeros(self.factors[i].shape),
                    scale=self.rank_parameters[i]),
                                reinterpreted_batch_ndims=4))

        self.factor_prior_distributions.append(
            td.Independent(td.Normal(
                loc=torch.zeros(self.factors[-1].shape),
                scale=self.rank_parameters[-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)),
                            reinterpreted_batch_ndims=4))

    def sample_full(self):
        return self.full_from_factors(
            [x.rsample() for x in self.factor_distributions])

    def sample_factors(self):
        
        return [x.rsample() for x in self.factor_distributions]

    def get_rank_parameters_update(self):

        updates = []

        for i in range(self.num_cores - 1):

            M = torch.sum(torch.square(self.factor_distributions[i].mean) +
                              torch.square(self.factor_distributions[i].stddev),
                              axis=[0,1, 2])
            if i == len(self.dims) - 2:
                D = self.dims[0][i]*self.dims[1][i]*self.rank_pairs[i][0]+self.dims[0][i+1]*self.dims[1][i+1]
                M += torch.sum(
                    torch.square(self.factor_distributions[i + 1].mean) +
                    torch.square(self.factor_distributions[i + 1].stddev),
                    axis=[1, 2, 3])
            else:
                D = self.dims[0][i]*self.dims[1][i]*self.rank_pairs[i][0]

            if self.prior_type == 'log_uniform':
                update = M / (D + 1)
            elif self.prior_type == 'gamma':
                update = (2 * self.beta + M) / (D + 2 + 2 * self.alpha)

            elif self.prior_type == 'half_cauchy':
                update = (M - (self.eta**2) * D +
                          torch.sqrt(M**2 + (M * self.eta**2) * (2.0 * D + 8.0) +
                                  (D**2.0) * (self.eta**4.0))) / (2 * D + 4.0)

            updates.append(update)

        return updates

    def update_rank_parameters(self):

        with torch.no_grad():

            rank_updates = self.get_rank_parameters_update()
            for rank_parameter, rank_update in zip(self.rank_parameters, rank_updates):
                
                sqrt_parameter_update = torch.sqrt((1 - self.em_stepsize) * rank_parameter.data**2 + self.em_stepsize * rank_update)
                rank_parameter.data.sub_(rank_parameter.data)
                rank_parameter.data.add_(sqrt_parameter_update.to(rank_parameter.device))

    def get_rank(self, threshold=1e-4):
        return [int(sum(torch.square(x) > threshold)) for x in self.rank_parameters]

    def get_kl_divergence_to_prior(self):

        kl_sum = 0.0

        for p,rank_parameter in zip(self.factor_distributions,self.rank_parameters+[self.rank_parameters[-1].unsqueeze(1).unsqueeze(2).unsqueeze(3)]):
            var_ratio = (p.stddev / rank_parameter).pow(2)
            t1 = ((p.mean ) / rank_parameter).pow(2)
            kl = torch.sum(0.5 * (var_ratio + t1 - 1 - var_ratio.log()))
            kl_sum+=kl


        return kl_sum

    def tensor_times_matrix(self,matrix_b):
    
        ndims = len(self.dims1)

        a_columns = self.shape[0]
        b_rows = matrix_b.get_shape()[0]

        a_shape = self.shape
        a_raw_shape = self.dims

        b_shape = matrix_b.shape

        a_ranks = [x[0] for x in self.rank_pairs]
        # If A is (i0, ..., id-1) x (j0, ..., jd-1) and B is (j0, ..., jd-1) x K,
        # data is (K, j0, ..., jd-2) x jd-1 x 1
        data = torch.transpose(matrix_b)
        data = torch.reshape(data, (-1, a_raw_shape[1][-1], 1))

        for core_idx in reversed(range(ndims)):
            curr_core = self.factors[core_idx]
            # On the k = core_idx iteration, after applying einsum the shape of data
            # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
            data = torch.einsum('aijb,rjb->ira', curr_core, data)
            if core_idx > 0:
            # After reshape the shape of data becomes
            # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
                new_data_shape = (-1, a_raw_shape[1][core_idx - 1], a_ranks[core_idx])
                data = torch.reshape(data, new_data_shape)
        # At the end the shape of the data is (i0, ..., id-1) x K
        out =  torch.reshape(data, (a_shape[0], b_shape[1]))
        return out
