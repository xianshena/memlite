"""~308M non-language relation/choice network; GPU execution is mandatory.

Typed numerical body, cue, world-context and candidate tokens. No tokenizer,
text generation, candidate-position embedding, or access to evaluator fields.
The large capacity is an experimental user-selected size, not an intelligence
claim. Each new personal event is written to the external experience table.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LargeChoiceConfig:
    width: int = 1024
    ff_width: int = 2816
    depth: int = 24
    heads: int = 16
    dropout: float = .1
    feature_count: int = 40
    train_temperature: float = .35


class RelationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config.width, config.ff_width
        self.heads = config.heads
        self.norm_attention = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.attention_output = nn.Linear(d, d, bias=False)
        self.norm_ff = nn.LayerNorm(d)
        self.up_gate = nn.Linear(d, 2*h, bias=False)
        self.down = nn.Linear(h, d, bias=False)
        self.scale_attention = nn.Parameter(torch.full((d,), .1))
        self.scale_ff = nn.Parameter(torch.full((d,), .1))
        self.dropout = config.dropout

    def forward(self, x, valid):
        batch, count, width = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(batch, count, 3, self.heads, width//self.heads)
        q, k, v = qkv.unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                                   dropout_p=0., is_causal=False)
        attention = attention.transpose(1, 2).reshape(batch, count, width)
        update = self.attention_output(attention) * self.scale_attention
        x = x + F.dropout(update, self.dropout, self.training)
        gate, up = self.up_gate(self.norm_ff(x)).chunk(2, -1)
        update = self.down(F.silu(gate)*up) * self.scale_ff
        return x + F.dropout(update, self.dropout, self.training)


class LargeChoiceNetwork(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LargeChoiceConfig()
        c = self.config
        if c.feature_count != 40 or c.width % c.heads:
            raise ValueError('40-column interface and integral attention heads required')
        self.body_encoder = nn.Linear(6, c.width)
        self.cue_encoder = nn.Linear(6, c.width)
        self.world_encoder = nn.Linear(8, c.width)
        self.candidate_encoder = nn.Linear(20, c.width)
        self.token_type = nn.Parameter(torch.randn(4, c.width)*.02)
        self.blocks = nn.ModuleList(RelationBlock(c) for _ in range(c.depth))
        self.norm = nn.LayerNorm(c.width)
        self.utility_head = nn.Linear(c.width, 1)
        self.outcome_head = nn.Linear(c.width, 6)
        self.register_buffer('feature_mean', torch.zeros(40))
        self.register_buffer('feature_scale', torch.ones(40))

    def forward(self, features, mask, temperature=None):
        if features.device.type != 'cuda' or mask.device.type != 'cuda':
            raise RuntimeError('CPU model execution prohibited by user; CUDA is required')
        if features.ndim != 3 or features.shape[1:] != (12, 40) or mask.shape != features.shape[:2]:
            raise ValueError('expected [batch,12,40] and [batch,12]')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('no legal candidate')
        if (not torch.equal(features[:, :, :17], features[:, :1, :17].expand(-1, 12, -1))
                or not torch.equal(features[:, :, 35:38], features[:, :1, 35:38].expand(-1, 12, -1))):
            raise ValueError('global context must be identical across candidate rows')
        x = (features-self.feature_mean)/self.feature_scale
        global_x = x[:, 0]
        world = torch.cat((global_x[:, 12:17], global_x[:, 35:38]), dim=-1)
        context = torch.stack((self.body_encoder(global_x[:, :6]),
                               self.cue_encoder(global_x[:, 6:12]), self.world_encoder(world)), dim=1)
        context = context + self.token_type[:3]
        candidate_x = torch.cat((x[:, :, 17:35], x[:, :, 38:40]), dim=-1)
        candidates = self.candidate_encoder(candidate_x) + self.token_type[3]
        tokens = torch.cat((context, candidates), dim=1)
        valid = torch.cat((torch.ones(features.shape[0], 3, dtype=torch.bool, device=features.device), mask), dim=1)
        for block in self.blocks:
            tokens = block(tokens, valid)
        candidates = self.norm(tokens[:, 3:])
        utilities = self.utility_head(candidates).squeeze(-1).float()
        outcomes = self.outcome_head(candidates).float()
        temp = self.config.train_temperature if temperature is None else temperature
        logits = (utilities/max(float(temp), 1e-4)).masked_fill(~mask, -1e9)
        return {'utilities': utilities, 'outcomes': outcomes, 'logits': logits,
                'probabilities': torch.softmax(logits, dim=-1)}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def teacher_distribution(utilities, mask):
    maxima = utilities.masked_fill(~mask, -1e9).amax(dim=-1, keepdim=True)
    best = ((utilities-maxima).abs() <= 1e-6) & mask
    return best.float()/best.sum(dim=-1, keepdim=True)


def choice_loss(prediction, mask, utilities, outcomes):
    target_p = teacher_distribution(utilities, mask)
    ce = -(target_p*torch.log_softmax(prediction['logits'], dim=-1)).sum(-1).mean()
    value = F.smooth_l1_loss(prediction['utilities'][mask], utilities[mask])
    outcome = F.smooth_l1_loss(prediction['outcomes'][mask], outcomes[mask])
    loss = ce + .25*value + .1*outcome
    return loss, {'choice': ce.detach().item(), 'value': value.detach().item(), 'outcome': outcome.detach().item()}
