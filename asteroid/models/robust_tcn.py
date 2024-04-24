import torch
from asteroid_filterbanks import make_enc_dec
import asteroid_filterbanks.transforms as trans
from ..masknn import TDConvNet
from .base_models import BaseEncoderMaskerDecoder
import warnings


class RobustTCN(BaseEncoderMaskerDecoder):
    """TCN masker a la ConvTasNet built on robust features.

    Args:
        n_src (int): Number of sources in the input mixtures.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        mask_input (str): Input representation to mask network. Choose from
            [``reim``, ``mag``, ``magreim``]
        mask_output (str): Type of mask to produce. Choose from [``reim``,
            ``mag``]
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
          for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
          https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        n_src,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="sigmoid",
        causal=False,
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        mask_input="reim",
        mask_output="reim",
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )

        if mask_input == "mag":
            self.pre_mask_fn = trans.mag
            n_feats = encoder.n_feats_out // 2
        elif mask_input == "magreim":
            self.pre_mask_fn = magreim
            n_feats = encoder.n_feats_out + encoder.n_feats_out // 2
        else:
            assert mask_input == "reim", (
                f"mask_input ('{mask_input}') not in ['mag', 'reim', 'magreim']"
            )
            self.pre_mask_fn = lambda x: x
            n_feats = encoder.n_feats_out

        if mask_output == "mag":
            self.mask_fn = trans.apply_mag_mask
            out_chan = encoder.n_feats_out // 2
        else:
            assert mask_output == "reim", (
                f"mask_output ('{mask_output}') not in ['mag', 'reim']"
            )
            self.mask_fn = trans.apply_complex_mask
            out_chan = encoder.n_feats_out

        if causal and norm_type not in ["cgLN", "cLN"]:
            norm_type = "cLN"
            warnings.warn(
                "In causal configuration cumulative layer normalization (cgLN)"
                "or channel-wise layer normalization (chanLN)  "
                f"must be used. Changing {norm_type} to cLN"
            )
        # Update in_chan
        masker = TDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        return self.masker(self.pre_mask_fn(tf_rep))

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        return self.mask_fn(tf_rep.unsqueeze(1), est_masks)
