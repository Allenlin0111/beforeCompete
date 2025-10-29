import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNEmbedPrefilter(nn.Module):
    """
    Multi-stage depthwise temporal prefilter with zero-phase response.

    Stage 1 parameters are configurable; stages 2 and 3 (optional) adopt wider
    kernels to cover longer temporal scales. Each stage preserves channel
    independence and starts as an identity map via impulse-initialised filters
    and zero-initialised gates.
    """

    def __init__(
        self,
        seq_len: int,
        c_out: int,
        kernel1: int = 3,
        dilation1: int = 1,
        kernel2: int = 5,
        dilation2: int = 1,
        kernel3: int = 7,
        dilation3: int = 2,
        stack: int = 1,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        if kernel1 % 2 == 0 or kernel2 % 2 == 0 or kernel3 % 2 == 0:
            raise ValueError("TCNEmbedPrefilter expects odd kernel sizes for symmetric padding.")

        self.seq_len = seq_len
        self.c_out = c_out
        self.stack = max(1, int(stack))
        self.use_gate = bool(use_gate)
        self.register_buffer("_eps", torch.tensor(1e-6), persistent=False)

        # Stage 1
        self.conv1 = nn.Conv1d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=kernel1,
            padding=0,
            dilation=dilation1,
            groups=c_out,
            bias=False,
        )
        self._init_as_identity(self.conv1.weight, kernel1)
        if self.use_gate:
            self.alpha1 = nn.Parameter(torch.zeros(1, 1, c_out))
        else:
            self.register_buffer("alpha1", torch.ones(1, 1, c_out), persistent=False)

        # Stage 2 (optional)
        if self.stack > 1:
            self.conv2 = nn.Conv1d(
                in_channels=c_out,
                out_channels=c_out,
                kernel_size=kernel2,
                padding=0,
                dilation=dilation2,
                groups=c_out,
                bias=False,
            )
            self._init_as_identity(self.conv2.weight, kernel2)
            if self.use_gate:
                self.alpha2 = nn.Parameter(torch.zeros(1, 1, c_out))
            else:
                self.register_buffer("alpha2", torch.ones(1, 1, c_out), persistent=False)
        else:
            self.conv2 = None
            self.register_buffer("alpha2", torch.zeros(1, 1, c_out), persistent=False)

        if self.stack > 2:
            self.conv3 = nn.Conv1d(
                in_channels=c_out,
                out_channels=c_out,
                kernel_size=kernel3,
                padding=0,
                dilation=dilation3,
                groups=c_out,
                bias=False,
            )
            self._init_as_identity(self.conv3.weight, kernel3)
            if self.use_gate:
                self.alpha3 = nn.Parameter(torch.zeros(1, 1, c_out))
            else:
                self.register_buffer("alpha3", torch.ones(1, 1, c_out), persistent=False)
        else:
            self.conv3 = None
            self.register_buffer("alpha3", torch.zeros(1, 1, c_out), persistent=False)

    @staticmethod
    def _init_as_identity(weight: torch.Tensor, kernel: int) -> None:
        with torch.no_grad():
            weight.zero_()
            center = (kernel - 1) // 2
            for c in range(weight.size(0)):
                weight[c, 0, center] = 1.0

    def _effective_kernel(self, weight: torch.Tensor) -> torch.Tensor:
        w = 0.5 * (weight + torch.flip(weight, dims=[-1]))
        w = w / (w.sum(dim=-1, keepdim=True) + self._eps)
        return w

    def _zero_phase_depthwise(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        # x: (B, L, C)
        x_ch = x.permute(0, 2, 1)  # (B, C, L)
        kernel = conv.kernel_size[0]
        dilation = conv.dilation[0]
        pad = (kernel - 1) * dilation
        weight = self._effective_kernel(conv.weight)

        x_pad = F.pad(x_ch, (pad, pad), mode="reflect")
        y = F.conv1d(x_pad, weight, dilation=dilation, groups=self.c_out)

        y_rev = torch.flip(y, dims=[-1])
        y_rev_pad = F.pad(y_rev, (pad, pad), mode="reflect")
        y2 = F.conv1d(y_rev_pad, weight, dilation=dilation, groups=self.c_out)
        y2 = torch.flip(y2, dims=[-1])
        # crop back to original sequence length
        L = x_ch.size(-1)
        start = pad
        end = start + L
        y2 = y2[..., start:end]

        return y2.permute(0, 2, 1)  # (B, L, C)

    def _apply_stage(self, x: torch.Tensor, conv: nn.Conv1d, alpha_param: torch.Tensor) -> torch.Tensor:
        filtered = self._zero_phase_depthwise(x, conv)
        if self.use_gate:
            gate = torch.sigmoid(alpha_param)
        else:
            gate = alpha_param
        return x + gate * (filtered - x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._apply_stage(x, self.conv1, self.alpha1)
        if self.stack > 1 and self.conv2 is not None:
            y = self._apply_stage(y, self.conv2, self.alpha2)
        if self.stack > 2 and self.conv3 is not None:
            y = self._apply_stage(y, self.conv3, self.alpha3)
        return y
