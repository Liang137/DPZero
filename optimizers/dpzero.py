import numpy as np

import torch
from torch.nn import CrossEntropyLoss


class DPZero():

    def __init__(self, model, lr, lam, clip, noise_multiplier, clip_scalar=True):
        self.model = model
        self.lr = lr
        self.lam = lam
        self.clip = clip
        self.noise_multiplier = noise_multiplier
        self.clip_scalar = clip_scalar

        if noise_multiplier == 0:
            self.clip_scalar = False


    def perturb_model(self, scaling_factor=1):
        torch.manual_seed(self.seed)

        for _, param in self.model.named_parameters():
            if param.requires_grad:
                u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * u * self.lam


    @torch.no_grad()
    def zo_forward(self, inputs):
        self.model.eval()
        outputs = self.model(**inputs)
        logits = outputs.logits

        # 'none' reduction to retain per sample loss
        loss_fct = CrossEntropyLoss(reduction='none')
        per_sample_loss = loss_fct(logits, inputs['labels'])
        return per_sample_loss.detach()


    @torch.no_grad()
    def zo_update(self, finite_diff):
        torch.manual_seed(self.seed)

        # perform per-sample clipping on the finite difference
        if self.clip_scalar:
            mask = finite_diff.abs() > self.clip
            reweight = torch.sign(finite_diff) * self.clip
            finite_diff[mask] = reweight[mask]
        
        # add DP noise to the finite difference
        if self.noise_multiplier != 0:
            # numpy use different seeds from torch
            noise = np.random.normal(loc=0., scale=self.noise_multiplier*self.clip, size=finite_diff.size()).mean()
        else:
            noise = 0

        for _, param in self.model.named_parameters():
            if param.requires_grad:
                u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data - self.lr * (finite_diff.mean() + noise) * u


    def step(self, inputs):
        # fix seed within each step
        self.seed = np.random.randint(1000000000)

        # x + \lam * u
        self.perturb_model(scaling_factor=1)
        loss_1 = self.zo_forward(inputs)

        # x - \lam * u
        self.perturb_model(scaling_factor=-2)
        loss_2 = self.zo_forward(inputs)

        finite_diff = (loss_1 - loss_2) / (2 * self.lam)

        # reset to x
        self.perturb_model(scaling_factor=1)
        self.zo_update(finite_diff)
