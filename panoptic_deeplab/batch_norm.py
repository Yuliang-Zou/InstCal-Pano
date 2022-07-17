# Implement different variants of InstCal
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class InstCalU(nn.Module):
    def __init__(self, num_features, eps=1e-5, per_channel=True, separate=True, init=0.1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.per_channel = per_channel
        self.separate = separate
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

        if per_channel:
            self.momentum = Parameter(init*torch.ones(num_features, **factory_kwargs))
        else:
            self.momentum = Parameter(torch.tensor(init, **factory_kwargs))

        if separate:
            if per_channel:
                self.momentum2 = Parameter(init*torch.ones(num_features, **factory_kwargs))
            else:
                self.momentum2 = Parameter(torch.tensor(init, **factory_kwargs))

    def forward(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, (2, 3))    # BxC
            # If spatial size is 1x1, unbiased var will be NaN
            if x.shape[2]*x.shape[3] < 2:
                unbiased = False
            else:
                unbiased = True
            batch_var  = torch.var(x, (2, 3), unbiased=unbiased)    # BxC
            # NOTE: momentum should be [0, 1]
            self.momentum.data = torch.clamp(self.momentum.data, min=self.eps, max=1.-self.eps)
            if self.separate:
                self.momentum2.data = torch.clamp(self.momentum2.data, min=self.eps, max=1.-self.eps)

        curr_mean = (1-self.momentum)*self.running_mean + self.momentum*batch_mean
        var_momentum = self.momentum2 if self.separate else self.momentum
        curr_var  = (1-var_momentum)*self.running_var + var_momentum*batch_var

        bz = x.shape[0]
        scale = self.weight * (curr_var+self.eps).rsqrt()
        bias  = self.bias - curr_mean*scale
        scale = scale.reshape(bz, -1, 1, 1)
        bias  = bias.reshape(bz, -1, 1, 1)
        out_dtype = x.dtype

        return x*scale.to(out_dtype)+bias.to(out_dtype)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "InstCalU(num_features={}, eps={}, per_channel={})".format(self.num_features, self.eps, self.per_channel)

    @classmethod
    def convert_adaptive_batchnorm(cls, module, per_channel=True, separate=True, init=0.1, device=None):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features, per_channel=per_channel, separate=separate, init=init, device=device)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps

            if per_channel:
                res.momentum.data = res.momentum.clone().detach()*torch.ones(module.num_features)
            else:
                res.momentum.data = res.momentum.clone().detach()

        else:
            for name, child in module.named_children():
                new_child = cls.convert_adaptive_batchnorm(child, per_channel, separate, init=init, device=device)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class InstCalC(nn.Module):
    def __init__(self, num_features, eps=1e-5, dim=256, num_basis=8, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.dim = dim
        self.num_basis = num_basis
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

        # Directly learning input irrelevant basis: 1 x k x C
        self.momentum_basis_mean = Parameter(torch.normal(mean=0., std=0.05, size=(1, num_basis, num_features)))
        self.momentum_basis_var = Parameter(torch.normal(mean=0., std=0.05, size=(1, num_basis, num_features)))

        # Predict the momentum basis coefficients
        self.mean_net = nn.Sequential(
            nn.Linear(num_features*2, dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_basis, bias=True),
            nn.Softmax(dim=1),
        )

        self.var_net = nn.Sequential(
            nn.Linear(num_features*2, dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_basis, bias=True),
            nn.Softmax(dim=1),
        )

        # NOTE: Properly init to get a small init prediction
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)    # original

    def forward(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, (2, 3))    # B x C
            # If spatial size is 1x1, unbiased var will be NaN
            if x.shape[2]*x.shape[3] < 2:
                unbiased = False
            else:
                unbiased = True
            batch_var  = torch.var(x, (2, 3), unbiased=unbiased)    # BxC

        running_mean = self.running_mean.expand_as(batch_mean)
        running_var = self.running_var.expand_as(batch_var)

        concat_mean = torch.cat([batch_mean, running_mean], dim=1)
        momentum_basis_mean = self.mean_net(concat_mean)    # B x k
        concat_var = torch.cat([batch_var, running_var], dim=1)
        momentum_basis_var = self.var_net(concat_var)    # B x k

        bz = x.shape[0]
        momentum_basis_mean = momentum_basis_mean.view(bz, self.num_basis, 1)    # B x k x 1
        momentum_basis_var = momentum_basis_var.view(bz, self.num_basis, 1)    # B x k x 1

        momentum_residual_mean = torch.sum(momentum_basis_mean*self.momentum_basis_mean, dim=1)    # B x C
        momentum_residual_var = torch.sum(momentum_basis_var*self.momentum_basis_var, dim=1)    # B x C

        momentum_mean = torch.clamp(0.1+momentum_residual_mean, min=self.eps, max=1.-self.eps)
        momentum_var = torch.clamp(0.1+momentum_residual_var, min=self.eps, max=1.-self.eps)

        curr_mean = (1-momentum_mean)*self.running_mean + momentum_mean*batch_mean
        curr_var  = (1-momentum_var)*self.running_var + momentum_var*batch_var

        scale = self.weight * (curr_var+self.eps).rsqrt()
        bias  = self.bias - curr_mean*scale
        scale = scale.reshape(bz, -1, 1, 1)
        bias  = bias.reshape(bz, -1, 1, 1)
        out_dtype = x.dtype

        return x*scale.to(out_dtype)+bias.to(out_dtype)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "InstCalC(num_features={}, eps={}, dim={})".format(self.num_features, self.eps, self.dim)

    @classmethod
    def convert_conditional_batchnorm(cls, module, dim=256, num_basis=8):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features, dim=dim, num_basis=num_basis)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps

        else:
            for name, child in module.named_children():
                new_child = cls.convert_conditional_batchnorm(child, dim=dim, num_basis=num_basis)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res
