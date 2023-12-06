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


    def perturb_model(self, scaling_factor=1):
        torch.manual_seed(self.seed)

        for _, param in self.model.named_parameters():
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


    def zo_update(self, finite_diff):
        torch.manual_seed(self.seed)

        if self.clip_scalar:
            # perform per-sample clipping on the finite-difference term
            threshold = torch.ones_like(finite_diff) * self.clip
            mask = finite_diff.abs() > threshold
            reweight = finite_diff / finite_diff.abs()
            finite_diff[mask] = reweight[mask] * self.clip
        
        for _, param in self.model.named_parameters():
            u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # add DP noise to the finite difference
            noise = np.random.normal(loc=0., scale=self.noise_multiplier*self.clip)
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