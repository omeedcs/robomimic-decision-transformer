import math

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as pyd
from einops import rearrange
import numpy as np


from robomimic.models.distributions import TanhWrappedDistribution


class TanhTransform(pyd.transforms.Transform):
    # Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.99, 0.99))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    # Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class InputNorm(nn.Module):
    """
    Normalize an input sequence like in Time Series Transformers
    but with moving statistics so that data can change with the policy.
    """

    def __init__(self, dim, beta=1e-3, init_nu=10.0, skip: bool = False):
        super().__init__()
        self.skip = skip
        self.mu = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.nu = nn.Parameter(torch.ones(dim) * init_nu, requires_grad=False)
        self.beta = beta
        self._t = nn.Parameter(torch.ones((1,)), requires_grad=False)

    @property
    def sigma(self):
        return torch.nan_to_num(torch.sqrt(self.nu - self.mu**2) + 1e-5).clamp(
            1e-3, 1e6
        )

    def normalize_values(self, val):
        """
        Normalize the input with instability protection.

        This function has to normalize lots of elements that are
        not well distributed (terminal signals, rewards, some
        parts of the state).
        """
        if self.skip:
            return val
        sigma = self.sigma
        normalized = ((val - self.mu) / sigma).clamp(-1e4, 1e4)
        not_nan = ~torch.isnan(normalized)
        stable = (sigma > 0.01).expand_as(not_nan)
        use_norm = torch.logical_and(stable, not_nan)
        output = torch.where(use_norm, normalized, (val - torch.nan_to_num(self.mu)))
        return output

    def denormalize_values(self, val):
        if self.skip:
            return val
        sigma = self.sigma
        denormalized = (val * sigma) + self.mu
        stable = (sigma > 0.01).expand_as(denormalized)
        output = torch.where(stable, denormalized, (val + torch.nan_to_num(self.mu)))
        return output

    def masked_stats(self, val, pad_mask):
        # make sure the padding value doesn't impact statistics
        keep_mask = (~pad_mask).float()
        val_keep = val * keep_mask
        sum_ = val_keep.sum((0, 1))
        square_sum = (val_keep**2).sum((0, 1))
        total = keep_mask.sum((0, 1))
        mean = sum_ / total
        square_mean = square_sum / total
        return mean, square_mean

    def update_stats(self, val, pad_mask):
        self._t += 1
        old_sigma = self.sigma
        old_mu = self.mu
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        mean, square_mean = self.masked_stats(val, pad_mask)
        self.mu.data = (1.0 - beta_t) * self.mu + (beta_t * mean)
        self.nu.data = (1.0 - beta_t) * self.nu + (beta_t * square_mean)

    def forward(self, x, denormalize=False):
        if denormalize:
            return self.denormalize_values(x)
        else:
            return self.normalize_values(x)


class Agent(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        d_inp: int,
        d_emb: int,
        d_action: int,
        policy: str,
        normalize_input: bool = True,
        gmm_modes: int = 5,
        ignore_rtg: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_action = d_action
        self.normalizer = InputNorm(dim=d_inp, skip=not normalize_input)
        self.policy = policy
        self.gmm_modes = gmm_modes
        self.ignore_rtg = ignore_rtg

        if policy == "gaussian":
            act_dim = 2 * d_action
        elif policy == "gmm":
            act_dim = gmm_modes * 2 * d_action + gmm_modes
        self.action_output = nn.Linear(d_emb, act_dim)

    def make_action_dist(
        self, params: torch.Tensor, log_std_low=-5.0, log_std_high=2.0
    ):
        if self.policy == "gaussian":
            mu, log_std = params.chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            log_std = log_std_low + 0.5 * (log_std_high - log_std_low) * (log_std + 1)
            std = log_std.exp()
            dist = SquashedNormal(mu, std)
        elif self.policy == "gmm":
            idx = self.gmm_modes * self.d_action
            means = rearrange(
                params[:, :, :idx], "b l (m p) -> b l m p", m=self.gmm_modes
            )
            log_std = rearrange(
                params[:, :, idx : 2 * idx], "b l (m p) -> b l m p", m=self.gmm_modes
            )
            log_std = torch.tanh(log_std)
            log_std = log_std_low + 0.5 * (log_std_high - log_std_low) * (log_std + 1)
            std = log_std.exp()
            logits = params[:, :, 2 * idx :]
            comp = pyd.Independent(pyd.Normal(loc=means, scale=std), 1)
            mix = pyd.Categorical(logits=logits)
            dist = pyd.MixtureSameFamily(
                mixture_distribution=mix, component_distribution=comp
            )
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.0)
        return dist

    def get_action(
        self,
        state_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        sample_action: bool = True,
    ) -> np.ndarray:
        batch, length, _ = state_seq.shape
        pad_mask = torch.zeros((batch, length, 1)).bool().to(state_seq.device)
        dist = self(state_seq, pad_mask=pad_mask)
        if sample_action:
            action = dist.sample()
        else:
            action = dist.mean
        current_action = torch.take_along_dim(action, seq_lengths - 1, dim=1).squeeze(1)
        return current_action.cpu().numpy()

    def forward(self, state_seq, pad_mask):
        if self.training:
            self.normalizer.update_stats(state_seq, pad_mask=pad_mask)
        norm_state_seq = self.normalizer(state_seq)
        if self.ignore_rtg:
            # zero out RTG information
            norm_state_seq[..., -1] = 0.0
        emb = self.encoder(states=norm_state_seq, pad_mask=pad_mask)
        action_output = self.action_output(emb)
        dist = self.make_action_dist(action_output)
        return dist
