from torch import nn

from context_score_10crop_try import ContextScore10CropTryModel
from hd_dgtm_simple_try import SimpleGateHDDGTMixer


class ContextScoreSimple10CropTryModel(ContextScore10CropTryModel):
    """10-crop simplified variant: no prompt branch, optional static/uniform scale gate."""

    def __init__(
        self,
        *args,
        scale_gate_mode='static',
        dilations=(1, 2, 4, 8, 16),
        n_features=None,
        use_ffn=True,
        **kwargs,
    ):
        kwargs['score_combo'] = 'mlp_context'
        if n_features is None:
            n_features = kwargs.get('n_features')
        super().__init__(*args, n_features=n_features, dilations=dilations, **kwargs)
        self.encoder = SimpleGateHDDGTMixer(
            n_features,
            dilations=dilations,
            dropout=0.1,
            gate_mode=scale_gate_mode,
            use_ffn=use_ffn,
        )
        self.encoder_bypass = nn.LayerNorm(n_features)

    def _semantic_logits_crop(self, semantic_features, bs, ncrops, t):
        return semantic_features.new_zeros(bs, ncrops, t)
