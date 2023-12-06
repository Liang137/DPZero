import dataclasses
import argparse
from tqdm.auto import tqdm

import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class Data:
    beta_train: torch.Tensor
    beta_test: torch.Tensor
    Ar: torch.Tensor  # A^{1/2}.
    sensitivity: float
    g0: float

    def __post_init__(self):
        self.n_train, self.d = self.beta_train.size()
        self.n_test = self.beta_test.shape[0]


def make_data(
    betas=None,
    n_train=100000, n_test=100000, d=10, dmin=1, mu_beta=0.2, si_beta=0.1,
    mode="linear",
    g0=1.,
):
    if betas is None:
        beta_train, beta_test = make_beta(
            n_train=n_train, n_test=n_test, d=d, dmin=dmin, mu_beta=mu_beta, si_beta=si_beta
        )
    else:
        beta_train, beta_test = betas
        n_train, d = beta_train.size()
        n_test, _ = beta_test.size()

    if mode == "const":
        Ar = g0 * torch.arange(1, d + 1, device=device) ** 0.0
    elif mode == "sqrt":
        Ar = g0 * torch.arange(1, d + 1, device=device) ** -.5
    elif mode == "linear":
        Ar = g0 * torch.arange(1, d + 1, device=device) ** -1.
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sensitivity = 2 * g0 / n_train

    return Data(beta_train=beta_train, beta_test=beta_test, Ar=Ar, sensitivity=sensitivity, g0=g0)


def make_beta(n_train, n_test, d, dmin, mu_beta, si_beta):
    if d < dmin:
        raise ValueError(f"d < dmin")

    beta_train = mu_beta + torch.randn(size=(n_train, d), device=device) * si_beta
    beta_train[:, dmin:] = 0.  # Ensure init distance to opt is the same.

    beta_test = mu_beta + torch.randn(size=(n_test, d), device=device) * si_beta
    beta_test[:, dmin:] = 0.  # Same distribution as train.

    return beta_train, beta_test


def evaluate(data: Data, beta: torch.Tensor, metric='Grad'):
    """Compute loss 1 / n sum_i | A^{1/2} (beta - beta_i) |_2 for train and test."""

    def compute_loss(samples):
        res = data.Ar[None, :] * (beta - samples)  # (n, d).
        return res.norm(2, dim=1).mean(dim=0).item()
    
    def compute_grad(samples):
        res = data.Ar[None, :] * (beta - samples)  # (n, d).
        grad = data.Ar * (res / res.norm(2, dim=1, keepdim=True)).mean(dim=0)
        return grad.norm(2)

    if metric == 'Loss':
        return tuple(
            compute_loss(samples=samples)
            for samples in (data.beta_train, data.beta_test)
        )
    elif metric == 'Grad':
        return tuple(
            compute_grad(samples=samples)
            for samples in (data.beta_train, data.beta_test)
        )
    else:
        raise NotImplementedError


def train_one_step(data: Data, beta, lr, epsilon, delta, weight_decay):
    res = data.Ar[None, :] * (beta - data.beta_train)  # (n, d).
    grad = data.Ar * (res / res.norm(2, dim=1, keepdim=True)).mean(dim=0)

    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * data.sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta = beta - lr * (grad_priv + weight_decay * beta)
    return beta


#############
def train_one_step_zero_naive(data: Data, beta, lr, epsilon, delta, weight_decay, u, lam):
    res_1 = data.Ar[None, :] * (beta + lam * u - data.beta_train)
    loss_1 = res_1.norm(2, dim=1).mean(dim=0).item()
    res_2 = data.Ar[None, :] * (beta - lam * u - data.beta_train)
    loss_2 = res_2.norm(2, dim=1).mean(dim=0).item()
    grad = ((loss_1 - loss_2) / (2 * lam)) * u

    sensitivity = data.sensitivity * (torch.norm(u) ** 2)
    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta = beta - lr * (grad_priv + weight_decay * beta)
    return beta
#############


#############
def train_one_step_dpzero(data: Data, beta, lr, epsilon, delta, weight_decay, u, lam):
    res_1 = data.Ar[None, :] * (beta + lam * u - data.beta_train)
    loss_1 = res_1.norm(2, dim=1)
    res_2 = data.Ar[None, :] * (beta - lam * u - data.beta_train)
    loss_2 = res_2.norm(2, dim=1)

    finite_diff = (loss_1 - loss_2) / (2 * lam)
    threshold = torch.ones_like(finite_diff) * data.g0
    mask = finite_diff.abs() > threshold
    reweight = finite_diff / finite_diff.abs()
    finite_diff[mask] = reweight[mask] * data.g0

    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * data.sensitivity ** 2. / epsilon ** 2.
    grad_priv = finite_diff.mean(dim=0).item() + np.random.normal() * math.sqrt(gaussian_mechanism_variance)
    beta = beta - lr * (grad_priv * u + weight_decay * beta)
    return beta
#############


@torch.no_grad()
def train(method, data: Data, num_steps, lr, weight_decay, epsilon, delta, lam, writer, metric='Grad'):
    per_step_epsilon, per_step_delta = make_per_step_privacy_spending(
        target_epsilon=epsilon, target_delta=delta, num_steps=num_steps
    )

    beta = torch.zeros(size=(1, data.d,), device=device)
    beta_avg = beta.clone()
    progress_bar = tqdm(range(num_steps))

    for global_step in range(0, num_steps):
        tr_loss, te_loss = evaluate(data=data, beta=beta_avg, metric=metric)

        if method == 'DPGD':
            beta = train_one_step(
                data=data,
                beta=beta,
                lr=lr, weight_decay=weight_decay,
                epsilon=per_step_epsilon, delta=per_step_delta,
            )

        elif method == 'naive':
            u = torch.randn_like(beta)
            beta = train_one_step_zero_naive(
                data=data,
                beta=beta,
                lr=lr, weight_decay=weight_decay,
                epsilon=per_step_epsilon, delta=per_step_delta,
                u=u, lam=lam,
            )

        elif method == 'DPZero':
            u = torch.randn_like(beta)
            beta = train_one_step_dpzero(
                data=data,
                beta=beta,
                lr=lr, weight_decay=weight_decay,
                epsilon=per_step_epsilon, delta=per_step_delta,
                u=u, lam=lam,
            )

        else:
            raise NotImplementedError

        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)
        writer.add_scalar('Train {}'.format(metric), tr_loss, global_step)
        writer.add_scalar('Test {}'.format(metric), te_loss, global_step)
        progress_bar.update(1)

    final_tr_loss, final_te_loss = evaluate(data=data, beta=beta_avg, metric=metric)
    writer.add_scalar('Train {}'.format(metric), final_tr_loss, num_steps)
    writer.add_scalar('Test {}'.format(metric), final_te_loss, num_steps)


def make_per_step_privacy_spending(
    target_epsilon, target_delta, num_steps, threshold=1e-4,
):
    per_step_delta = target_delta / (num_steps + 1)

    def adv_composition(per_step_epsilon):
        total_epsilon = (
            math.sqrt(2 * num_steps * math.log(1 / per_step_delta)) * per_step_epsilon +
            num_steps * per_step_epsilon * (math.exp(per_step_epsilon) - 1)
        )
        return total_epsilon

    minval, maxval = 1e-6, 5
    while maxval - minval > threshold:
        midval = (maxval + minval) / 2
        eps = adv_composition(midval)
        if eps > target_epsilon:
            maxval = midval
        else:
            minval = midval
    per_step_epsilon = minval
    return per_step_epsilon, per_step_delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy experiments')
    parser.add_argument('--num_steps', default=1000, type=int, help='training steps')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lam', default=0.0001, type=float, help='smoothing parameter')
    parser.add_argument('--eps', default=1., type=float, help='epsilon')
    parser.add_argument('--delta', default=1e-6, type=float, help='delta')
    parser.add_argument('--method', default='DPZero', type=str, help='optimizer')

    parser.add_argument('--n_train', default=5000, type=int, help='number of training samples')
    parser.add_argument('--n_test', default=5000, type=int, help='number of test samples')
    parser.add_argument('--d', default=1000, type=int, help='dimension')
    parser.add_argument('--mode', default='linear', type=str, help='spectral of Hessian')
    parser.add_argument('--metric', default='Grad', type=str, help='metric for evaluation')
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    writer = SummaryWriter(comment="_mode={}_method={}_d={}_lr={}_metric={}".format(args.mode, args.method, args.d, args.lr, args.metric))

    betas = make_beta(n_train=args.n_train, n_test=args.n_test, d=args.d, dmin=1, mu_beta=1, si_beta=1)
    data = make_data(betas=betas, mode=args.mode, g0=3.)
    train(args.method, data, args.num_steps, args.lr, 0, args.eps, args.delta, args.lam, writer, args.metric)
    writer.close()