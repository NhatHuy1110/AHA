"""Linear-chain workflow CRF over surgical phases, exact and parallel in time.

Inference uses log-semiring prefix products (a parallel scan), so a multi-hour
video at 1 FPS costs O(log T) batched tensor operations instead of a T-step loop.
The quantity the decoder consumes is the posterior probability that phase p
*starts* at second t:  P(y[t-1] != p, y[t] = p | whole video).
"""
import torch
from torch import nn

NEG = -1e4  # finite stand-in for log(0); keeps gradients NaN-free

def _mul(a, b):
    """Log-semiring matrix product over the last two dimensions."""
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), -2)

def _prefix(m):
    """P[t] = m[0] (x) ... (x) m[t] for every t (Hillis-Steele scan)."""
    p, d = m, 1
    while d < len(m):
        p = torch.cat([p[:d], _mul(p[:-d], p[d:])], 0)
        d *= 2
    return p

def _product(m):
    """m[0] (x) ... (x) m[-1] by pairwise reduction; O(T) memory."""
    while len(m) > 1:
        tail = m[-1:] if len(m) % 2 else m[:0]
        m = torch.cat([_mul(m[0:len(m)-len(tail):2], m[1::2]), tail], 0)
    return m[0]

class WorkflowCRF(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.transition = nn.Parameter(torch.zeros(states, states))
        self.start = nn.Parameter(torch.zeros(states))

    def _potentials(self, emit):
        return self.start + emit[0], self.transition[None] + emit[1:, None, :]

    def log_partition(self, emit):
        a0, m = self._potentials(emit)
        if not len(m):
            return torch.logsumexp(a0, 0)
        return torch.logsumexp(a0[:, None] + _product(m), (0, 1))

    def nll(self, emit, labels):
        """Mean per-second negative log-likelihood; seconds labelled -1 are marginalised."""
        allowed = torch.ones_like(emit, dtype=torch.bool)
        known = labels >= 0
        allowed[known] = False
        allowed[known, labels[known]] = True
        constrained = emit.masked_fill(~allowed, NEG)
        return (self.log_partition(emit) - self.log_partition(constrained)) / len(emit)

    def posteriors(self, emit):
        """Returns (log phase marginals [T,K], log onset posteriors [T,K])."""
        a0, m = self._potentials(emit)
        k = emit.shape[1]
        if not len(m):
            return torch.log_softmax(a0, 0)[None], emit.new_full((1, k), NEG)
        alpha = torch.cat([a0[None], torch.logsumexp(a0[None, :, None] + _prefix(m), 1)], 0)
        suffix = _prefix(m.flip(0).transpose(1, 2)).transpose(1, 2).flip(0)  # m[t] (x) ... (x) m[-1]
        beta = torch.cat([torch.logsumexp(suffix, 2), emit.new_zeros(1, k)], 0)
        log_z = torch.logsumexp(alpha[-1], 0)
        xi = alpha[:-1, :, None] + m + beta[1:, None, :] - log_z  # [T-1, from, to]
        entering = xi.masked_fill(torch.eye(k, dtype=torch.bool, device=emit.device)[None], NEG)
        onset = torch.cat([emit.new_full((1, k), NEG), torch.logsumexp(entering, 1)], 0)
        return alpha + beta - log_z, onset.clamp_min(NEG)
