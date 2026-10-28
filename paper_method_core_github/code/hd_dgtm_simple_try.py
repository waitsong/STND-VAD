import torch
from torch import nn


class SimpleGateHDDGTMixer(nn.Module):
    def __init__(self, dim, dilations=(1, 2, 4, 8, 16), dropout=0.1, gate_mode='learned', use_ffn=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dilations = tuple(dilations)
        self.gate_mode = gate_mode
        self.use_ffn = use_ffn
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=dim,
                )
                for dilation in self.dilations
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, len(self.dilations)),
        )
        self.static_gate = nn.Parameter(torch.zeros(len(self.dilations)))
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.last_scale_features = None
        self.last_scale_gate = None

    def _gate(self, x_norm):
        if self.gate_mode == 'learned':
            return torch.softmax(self.scale_gate(x_norm), dim=-1)
        if self.gate_mode == 'uniform':
            shape = (*x_norm.shape[:2], len(self.dilations))
            return x_norm.new_full(shape, 1.0 / len(self.dilations))
        if self.gate_mode == 'static':
            gate = torch.softmax(self.static_gate, dim=0)
            return gate.view(1, 1, -1).expand(x_norm.size(0), x_norm.size(1), -1)
        raise ValueError(f'unknown gate_mode: {self.gate_mode}')

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        xt = x_norm.transpose(1, 2)
        branch_outs = [branch(xt).transpose(1, 2) for branch in self.branches]
        groups = torch.stack(branch_outs, dim=2)

        gate = self._gate(x_norm).unsqueeze(-1)
        self.last_scale_features = groups
        self.last_scale_gate = gate.squeeze(-1)
        mixed = torch.sum(groups * gate, dim=2)

        update = self.out_proj(mixed)
        x = self.out_norm(residual + residual * update)
        if self.use_ffn:
            x = self.out_norm(x + self.ffn(x))
        return x
