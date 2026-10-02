"""Dense answer-onset decoder: every second of the video is a citation candidate.

    frozen frame features [T,D]
        -> dilated temporal encoder            g[t]
        -> phase / tool / onset heads          (dense supervision on g)
        -> workflow CRF over phases            log P(phase p starts at t | video)
        -> question/option cross-attention     h[q,t]
        -> s(t,a) = alpha * onset evidence(t | question)
                  + beta  * tool evidence(a | t)
                  + residual pair score(h[q,t], option a, question)

The scorer shares all weights across options and has no option-position input,
so it is permutation-equivariant in the options.
"""
import torch
import torch.nn.functional as F
from torch import nn
from .crf import WorkflowCRF

class DilatedLayer(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(width, width, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return x + self.drop(self.out(F.gelu(self.conv(x))))

class FusionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width*4, width))
    def forward(self, x, context):
        x = self.norm1(x + self.attn(x, context, context, need_weights=False)[0])
        return self.norm2(x + self.ff(x))

class AnswerOnsetDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config['width']
        self.variant = config['variant']          # joint | independent
        self.workflow = config['workflow']        # crf | head
        self.structured = config['structured']    # explicit onset/tool evidence terms
        self.residual = config['residual']        # neural pair term
        if self.variant not in ('joint', 'independent') or self.workflow not in ('crf', 'head'):
            raise ValueError('Unknown model variant')
        if not (self.structured or self.residual):
            raise ValueError('Scorer needs a structured or a residual term')
        self.in_drop = nn.Dropout(config['feature_dropout'])
        self.proj = nn.Linear(config['visual_dim'], d)
        self.position = nn.Sequential(nn.Linear(1, d), nn.Tanh())
        self.encoder = nn.ModuleList([DilatedLayer(d, 2**i, config['dropout']) for i in range(config['temporal_layers'])])
        self.norm = nn.LayerNorm(d)
        self.phase = nn.Linear(d, 12)
        self.tool = nn.Linear(d, 8)
        self.onset = nn.Linear(d, 12)
        nn.init.constant_(self.onset.bias, -4.6)  # onsets are rare: start near p=0.01
        self.crf = WorkflowCRF(12) if self.workflow == 'crf' else None
        self.text = nn.Linear(config['text_dim'], d)
        self.question_phase = nn.Linear(d, 12)
        self.option_tool = nn.Linear(d, 8)
        self.fusion = nn.ModuleList([FusionBlock(d, config['heads'], config['dropout']) for _ in range(config['fusion_layers'])])
        self.pair = nn.Sequential(nn.Linear(d*4, d), nn.GELU(), nn.Dropout(config['dropout']), nn.Linear(d, 1))
        self.time_head = nn.Linear(d, 1)
        self.scale = nn.Parameter(torch.full((2,), 0.5413))  # softplus -> 1.0

    def encode(self, x, position):
        """Question-independent video pass. x [T,D], position [T,1]."""
        g = self.proj(self.in_drop(x)) + self.position(position)
        g = g.t()[None]
        for layer in self.encoder:
            g = layer(g)
        g = self.norm(g[0].t())
        out = dict(g=g, phase=self.phase(g), tool=self.tool(g), onset=self.onset(g))
        out['emit'] = torch.log_softmax(out['phase'], -1)
        if self.crf is not None:
            _, out['onset_logpost'] = self.crf.posteriors(out['emit'])
        else:
            out['onset_logpost'] = F.logsigmoid(out['onset'])
        # Tool evidence for "visible at onset t": mean logit over [t, t+window).
        w = self.config['tool_window']
        padded = F.pad(out['tool'].t()[None], (0, w-1), mode='replicate')
        out['tool_evidence'] = F.logsigmoid(F.avg_pool1d(padded, w, stride=1)[0].t())
        # Floor the structured evidence so the residual term can still recover a missed onset.
        floor = self.config['evidence_floor']
        out['onset_logpost'] = out['onset_logpost'].clamp_min(floor)
        out['tool_evidence'] = out['tool_evidence'].clamp_min(floor)
        return out

    def forward(self, x, position, question, options):
        """question [Q,E], options [Q,A,E] -> logits [Q,T,A] plus dense outputs."""
        out = self.encode(x, position)
        q = self.text(question)                      # [Q,d]
        o = self.text(options)                       # [Q,A,d]
        nq, na = o.shape[:2]
        log_pi = torch.log_softmax(self.question_phase(F.gelu(q)), -1)       # [Q,12]
        log_rho = torch.log_softmax(self.option_tool(F.gelu(o)), -1)         # [Q,A,8]
        h = out['g'][None].expand(nq, -1, -1)
        context = torch.cat([q[:, None], o], 1)
        for layer in self.fusion:
            h = layer(h, context)
        alpha, beta = F.softplus(self.scale)
        onset_ev = torch.logsumexp(log_pi[:, None, :] + out['onset_logpost'][None], -1)            # [Q,T]
        tool_ev = torch.logsumexp(log_rho[:, None] + out['tool_evidence'][None, :, None, :], -1)   # [Q,T,A]
        nt = h.shape[1]
        if self.variant == 'independent':
            u = self.time_head(h).squeeze(-1)
            if self.structured:
                u = u + alpha*onset_ev
            pooled = (u.softmax(1)[:, :, None]*h).sum(1)[:, None].expand(-1, na, -1)
            v = self.pair(torch.cat([pooled, o, pooled*o, q[:, None].expand(-1, na, -1)], -1)).squeeze(-1)
            logits = u[:, :, None] + v[:, None, :]   # factorizes exactly
        else:
            logits = h.new_zeros(nq, nt, na)
            if self.structured:
                logits = logits + alpha*onset_ev[:, :, None] + beta*tool_ev
            if self.residual:
                qe = q[:, None].expand(-1, nt, -1)
                cols = []
                for a in range(na):  # one option at a time bounds memory on multi-hour videos
                    oa = o[:, a][:, None].expand(-1, nt, -1)
                    cols.append(self.pair(torch.cat([h, oa, h*oa, qe], -1)).squeeze(-1))
                logits = logits + torch.stack(cols, -1)
        out.update(logits=logits, log_pi=log_pi, log_rho=log_rho, onset_evidence=onset_ev, tool_evidence_options=tool_ev)
        return out
