# Ultralytics YOLO slimmable modules for early-exit experiments.
# Place this file at: ultralytics/nn/modules/slim.py

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import autopad, Conv
from .block import DFL
from .head import Detect

SLIM_WIDTHS = (1.0, 0.75, 0.5, 0.25)


def _width_key(width: float) -> str:
    # Restrict v0 to known discrete widths so SwitchableBN is deterministic.
    width = float(width)
    closest = min(SLIM_WIDTHS, key=lambda w: abs(w - width))
    if abs(closest - width) > 1e-6:
        raise ValueError(f"Unsupported width {width}. Use one of {SLIM_WIDTHS}.")

    # ModuleDict keys cannot contain "."
    return "w" + str(closest).replace(".", "_")


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    return max(min_value, int(math.ceil(v / divisor) * divisor))


def _active(c_max: int, width: float) -> int:
    # Never exceed max; small layers should not round above c_max.
    return min(c_max, _make_divisible(c_max * float(width), 8))

def _kernel_pair(k):
    if isinstance(k, int):
        return k, k
    if isinstance(k, tuple):
        if len(k) == 2 and all(isinstance(v, int) for v in k):
            return k
    raise TypeError(f"Unsupported kernel size: {k}")

class SwitchableBN2d(nn.Module):
    """Separate BN statistics for each discrete width."""

    def __init__(self, c_max: int, widths=SLIM_WIDTHS):
        super().__init__()
        self.c_max = int(c_max)
        self.widths = tuple(float(w) for w in widths)
        self.width_mult = 1.0
        self.bns = nn.ModuleDict({
            _width_key(w): nn.BatchNorm2d(_active(self.c_max, w))
            for w in self.widths
        })

    def set_width(self, width: float):
        self.width_mult = float(width)
        _width_key(self.width_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bns[_width_key(self.width_mult)](x)


class SlimConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c1_max = int(c1)
        self.c2_max = int(c2)

        kh, kw = _kernel_pair(k)
        self.k = (kh, kw)
        self.s = s
        self.g = g
        self.d = d

        if p is None:
            self.p = (kh // 2, kw // 2)
        else:
            self.p = p

        self.width_mult = 1.0

        self.weight = nn.Parameter(torch.empty(self.c2_max, self.c1_max, kh, kw))
        self.bn = SwitchableBN2d(self.c2_max)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def set_width(self, width: float):
        self.width_mult = float(width)
        self.bn.set_width(width)

    def active_out(self):
        return _active(self.c2_max, self.width_mult)

    def forward(self, x):
        cin = x.shape[1]
        cout = self.active_out()

        w = self.weight[:cout, :cin, :, :]

        return self.act(
            self.bn(
                F.conv2d(
                    x,
                    w,
                    bias=None,
                    stride=self.s,
                    padding=self.p,
                    dilation=self.d,
                    groups=1,
                )
            )
        )


class SlimPredConv(nn.Module):
    """1x1 prediction conv with sliced input channels and fixed output channels."""

    def __init__(self, c1_max: int, c2: int):
        super().__init__()
        self.c1_max = int(c1_max)
        self.c2 = int(c2)
        self.conv = nn.Conv2d(self.c1_max, self.c2, 1, bias=True)

    def set_width(self, width: float):
        # No output-width change; prediction dimensionality must stay fixed.
        return None

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cin = x.shape[1]
        w = self.conv.weight[:, :cin]
        return F.conv2d(x, w, self.conv.bias, 1, 0)


class SlimBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        k1 = k[0] if isinstance(k, tuple) and len(k) == 2 else 3
        k2 = k[1] if isinstance(k, tuple) and len(k) == 2 else 3

        self.cv1 = SlimConv(c1, c_, k1, 1)
        self.cv2 = SlimConv(c_, c2, k2, 1)
        self.add = shortcut and c1 == c2

    def set_width(self, width: float):
        self.cv1.set_width(width)
        self.cv2.set_width(width)

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add and x.shape == y.shape else y

class SlimC2f(nn.Module):
    """Slimmable version of Ultralytics C2f."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = SlimConv(c1, 2 * self.c, 1, 1)
        self.cv2 = SlimConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            SlimBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def set_width(self, width: float):
        self.cv1.set_width(width)
        self.cv2.set_width(width)
        for m in self.m:
            if hasattr(m, "set_width"):
                m.set_width(width)

    def forward(self, x):
        x = self.cv1(x)

        # Split by actual output channels to avoid rounding mismatches.
        c = x.shape[1] // 2
        y = [x[:, :c], x[:, c:]]

        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SlimC3k2(SlimC2f):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        attn=False,
        g=1,
        shortcut=True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)

        if attn:
            # Match stock C3k2 attn path more closely:
            # nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), PSABlock(...))
            # For now we only approximate the Bottleneck part, but e must be 0.5.
            self.m = nn.ModuleList(
                SlimBottleneck(self.c, self.c, shortcut, g, e=0.5)
                for _ in range(n)
            )
        elif c3k:
            # Stock C3k(self.c, self.c, 2, shortcut, g) uses e=0.5 by default.
            self.m = nn.ModuleList(
                SlimC3k(self.c, self.c, 2, shortcut, g, e=0.5)
                for _ in range(n)
            )
        else:
            # IMPORTANT: stock C3k2 uses Bottleneck(...), whose default e is 0.5.
            self.m = nn.ModuleList(
                SlimBottleneck(self.c, self.c, shortcut, g, e=0.5)
                for _ in range(n)
            )

    def set_width(self, width: float):
        self.width_mult = float(width)
        self.cv1.set_width(width)
        self.cv2.set_width(width)
        for m in self.m:
            m.set_width(width)

    def forward(self, x):
        x = self.cv1(x)

        # Split based on the actual output channels from cv1.
        # This avoids make_divisible rounding mismatches at widths like 0.75.
        c = x.shape[1] // 2
        y = [x[:, :c], x[:, c:]]

        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SlimSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k=5, n=3, shortcut=False):
        super().__init__()
        self.c = c1 // 2
        self.cv1 = SlimConv(c1, self.c, 1, 1, act=False)
        self.cv2 = SlimConv(self.c * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def set_width(self, width: float):
        self.cv1.set_width(width)
        self.cv2.set_width(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        out = self.cv2(torch.cat(y, 1))
        return out + x if self.add and out.shape == x.shape else out


class SlimC2PSA(nn.Module):
    """Slimmable placeholder for C2PSA.

    v0 intentionally uses bottlenecks instead of attention to prove the width plumbing.
    Replace internals later with slimmable PSA/attention once the base model runs.
    """

    def __init__(self, c1: int, c2: int, n=1, e=0.5):
        super().__init__()
        self.cv1 = SlimConv(c1, c2, 1, 1)
        self.m = nn.ModuleList(SlimBottleneck(c2, c2, shortcut=True, e=e) for _ in range(n))
        self.cv2 = SlimConv(c2, c2, 1, 1)

    def set_width(self, width: float):
        self.cv1.set_width(width)
        for m in self.m:
            m.set_width(width)
        self.cv2.set_width(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        for m in self.m:
            y = m(y)
        return self.cv2(y)


class SlimDetect(Detect):
    """Detect head that accepts reduced-width feature maps but keeps fixed prediction output."""

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):  # same signature as Detect
        nn.Module.__init__(self)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.dynamic = False
        self.export = False
        self.format = None
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.xyxy = False
        self.max_det = 300
        self.agnostic_nms = False
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(SlimConv(x, c2, 3), SlimConv(c2, c2, 3), SlimPredConv(c2, 4 * self.reg_max)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(SlimConv(x, c3, 3), SlimConv(c3, c3, 3), SlimPredConv(c3, self.nc)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.end2end = end2end
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def set_width(self, width: float):
        for module in self.modules():
            if module is not self and hasattr(module, "set_width"):
                module.set_width(width)

class SlimToFixed(nn.Module):
    """
    Converts slimmable feature channels back to fixed feature channels.

    Width behavior:
        width 1.0  -> use self.weight
        width 0.75 -> use self.weight_w0_75

    Also owns a tiny width-0.75 class-logit calibrator:
        class_scale_w0_75: [1, 80, 1]
        class_bias_w0_75:  [1, 80, 1]

    In this experiment only model.25's class calibration params are trained and
    used by the inference patch. The params live inside SlimToFixed so they are
    part of normal state_dict/checkpoint serialization.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, bias=False):
        super().__init__()

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.k = k if isinstance(k, tuple) else (k, k)
        self.s = s if isinstance(s, tuple) else (s, s)

        if p is None:
            self.p = (self.k[0] // 2, self.k[1] // 2)
        else:
            self.p = p if isinstance(p, tuple) else (p, p)

        self.width_mult = 1.0
        self.class_calib_nc = 80

        # Original/full-width bridge. Keep the name "weight" so old checkpoints
        # still load model.23.weight, model.24.weight, model.25.weight.
        self.weight = nn.Parameter(torch.empty(self.c2, self.c1, self.k[0], self.k[1]))

        # Width-specific bridge for 0.75.
        self.weight_w0_75 = nn.Parameter(torch.empty(self.c2, self.c1, self.k[0], self.k[1]))

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.c2))
            self.bias_w0_75 = nn.Parameter(torch.zeros(self.c2))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("bias_w0_75", None)

        # Width-specific class-logit calibration params. Identity initialized.
        self.class_scale_w0_75 = nn.Parameter(torch.ones(1, self.class_calib_nc, 1))
        self.class_bias_w0_75 = nn.Parameter(torch.zeros(1, self.class_calib_nc, 1))

        self.reset_parameters()

    def __setstate__(self, state):
        """Backfill attributes/parameters when loading old pickled checkpoints."""
        super().__setstate__(state)
        self._ensure_compat_attrs()

    def _ensure_compat_attrs(self):
        """Backfill attributes missing from older checkpoints."""
        if not hasattr(self, "width_mult"):
            self.width_mult = 1.0

        if not hasattr(self, "weight"):
            raise RuntimeError("SlimToFixed checkpoint is missing required parameter 'weight'.")

        if not hasattr(self, "c2"):
            self.c2 = int(self.weight.shape[0])
        if not hasattr(self, "c1"):
            self.c1 = int(self.weight.shape[1])
        if not hasattr(self, "k"):
            self.k = (int(self.weight.shape[2]), int(self.weight.shape[3]))
        if not hasattr(self, "s"):
            self.s = (1, 1)
        if not hasattr(self, "p"):
            self.p = (self.k[0] // 2, self.k[1] // 2)
        if not hasattr(self, "class_calib_nc"):
            self.class_calib_nc = 80

        if "bias" not in self._parameters:
            self.register_parameter("bias", None)
        if "bias_w0_75" not in self._parameters:
            self.register_parameter("bias_w0_75", None)

        if "weight_w0_75" not in self._parameters or self._parameters["weight_w0_75"] is None:
            self.register_parameter("weight_w0_75", nn.Parameter(self.weight.detach().clone()))

        ref = self.weight
        device = ref.device
        dtype = ref.dtype if ref.is_floating_point() else torch.float32

        if "class_scale_w0_75" not in self._parameters or self._parameters["class_scale_w0_75"] is None:
            self.register_parameter(
                "class_scale_w0_75",
                nn.Parameter(torch.ones(1, int(self.class_calib_nc), 1, device=device, dtype=dtype)),
            )
        if "class_bias_w0_75" not in self._parameters or self._parameters["class_bias_w0_75"] is None:
            self.register_parameter(
                "class_bias_w0_75",
                nn.Parameter(torch.zeros(1, int(self.class_calib_nc), 1, device=device, dtype=dtype)),
            )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight_w0_75.copy_(self.weight)
            self.class_scale_w0_75.fill_(1.0)
            self.class_bias_w0_75.zero_()

        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            with torch.no_grad():
                self.bias_w0_75.copy_(self.bias)

    def set_width(self, width_mult: float):
        self.width_mult = float(width_mult)
        return self

    def _active_weight_and_bias(self):
        self._ensure_compat_attrs()
        if abs(float(getattr(self, "width_mult", 1.0)) - 0.75) < 1e-6:
            return self.weight_w0_75, self.bias_w0_75
        return self.weight, self.bias

    def forward(self, x):
        self._ensure_compat_attrs()
        weight, bias = self._active_weight_and_bias()
        in_ch = x.shape[1]
        weight = weight[:, :in_ch, :, :]
        return F.conv2d(x, weight, bias, stride=self.s, padding=self.p)

class SlimAuxDetect(SlimDetect):
    """Auxiliary slimmable detection head for early exit."""
    pass

class SlimC3k(nn.Module):
    """Slimmable version of C3k: cv1/cv2/cv3 plus bottleneck stack."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = SlimConv(c1, c_, 1, 1)
        self.cv2 = SlimConv(c1, c_, 1, 1)
        self.cv3 = SlimConv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(
            *(SlimBottleneck(c_, c_, shortcut, g, k=((k, k), (k, k)), e=1.0) for _ in range(n))
        )

    def set_width(self, width: float):
        self.cv1.set_width(width)
        self.cv2.set_width(width)
        self.cv3.set_width(width)
        for m in self.m:
            if hasattr(m, "set_width"):
                m.set_width(width)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))