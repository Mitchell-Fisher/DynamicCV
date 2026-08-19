# Ultralytics YOLO slimmable modules for Yu-style slimmable networks.

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import autopad
from .block import DFL
from .head import Detect

DEFAULT_SLIM_WIDTHS = (
    1.0,
    0.875,
    0.75,
    0.625,
    0.5,
)

SLIM_WIDTHS = DEFAULT_SLIM_WIDTHS


def _normalize_slim_widths(widths):
    widths = tuple(float(w) for w in widths)

    if not widths:
        raise ValueError("At least one slimmable width is required.")

    if not any(abs(w - 1.0) < 1e-6 for w in widths):
        raise ValueError("Slimmable widths must include 1.0.")

    for w in widths:
        if not 0.0 < w <= 1.0:
            raise ValueError(
                f"Invalid slimmable width {w}. Widths must be in (0, 1]."
            )

    # Unique, largest -> smallest.
    return tuple(sorted(set(widths), reverse=True))


def configure_slim_widths(widths):
    global SLIM_WIDTHS

    SLIM_WIDTHS = _normalize_slim_widths(widths)

    return SLIM_WIDTHS


def _width_key(width: float, widths=None) -> str:
    width = float(width)

    widths = (
        SLIM_WIDTHS
        if widths is None
        else tuple(float(w) for w in widths)
    )

    closest = min(
        widths,
        key=lambda w: abs(w - width),
    )

    if abs(closest - width) > 1e-6:
        raise ValueError(
            f"Unsupported width {width}. "
            f"Use one of {widths}."
        )

    return "w" + str(closest).replace(".", "_")


def _active(c_max: int, width: float) -> int:
    c = int(round(c_max * float(width)))
    return max(1, min(c, c_max))

def _concat_prefix_indices(full_sizes, active_sizes, device):
    """
    Build full-width channel indices for a compact concatenation.

    Example:
        full groups   = [128, 256]
        active groups = [112, 224]

    returns:
        [0..111, 128..351]
    """
    if len(full_sizes) != len(active_sizes):
        raise ValueError(
            f"full_sizes has {len(full_sizes)} groups but "
            f"active_sizes has {len(active_sizes)} groups"
        )

    indices = []
    offset = 0

    for full_c, active_c in zip(full_sizes, active_sizes):
        full_c = int(full_c)
        active_c = int(active_c)

        if active_c > full_c:
            raise ValueError(
                f"Active channels {active_c} exceed full channels {full_c}"
            )

        indices.append(
            torch.arange(
                offset,
                offset + active_c,
                device=device,
                dtype=torch.long,
            )
        )

        offset += full_c

    if not indices:
        return torch.empty(0, device=device, dtype=torch.long)

    return torch.cat(indices)


def _group_prefix_indices(
    group_max,
    group_active,
    num_groups,
    device,
):
    """Convenience wrapper for equally sized logical channel groups."""
    return _concat_prefix_indices(
        [group_max] * num_groups,
        [group_active] * num_groups,
        device,
    )

class SwitchableBN2d(nn.BatchNorm2d):
    def __init__(self, c_max, widths=None):
        super().__init__(
            c_max,
            eps=0.001,
            momentum=0.03,
        )

        self.c_max = c_max
        self.width_mult = 1.0

        self.widths = _normalize_slim_widths(
            SLIM_WIDTHS if widths is None else widths
        )
        self.width_bns = nn.ModuleDict({
            _width_key(w, self.widths): nn.BatchNorm2d(
                _active(c_max, w),
                eps=0.001,
                momentum=0.03,
            )
            for w in self.widths
            if abs(w - 1.0) > 1e-6
        })

    def set_width(self, width):
        _width_key(width, self.widths)
        self.width_mult = float(width)

    def forward(self, x):
        if abs(self.width_mult - 1.0) < 1e-6:
            return super().forward(x)

        return self.width_bns[
            _width_key(self.width_mult, self.widths)
        ](x)

    def active_features(self):
        if abs(self.width_mult - 1.0) < 1e-6:
            return self.num_features

        return self.width_bns[
            _width_key(
                self.width_mult,
                self.widths,
            )
        ].num_features


class SlimConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.c1_max = int(c1)
        self.c2_max = int(c2)
        self.width_mult = 1.0
        self.base_groups = g

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )

        self.bn = SwitchableBN2d(c2)

        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def set_width(self, width: float):
        self.bn.set_width(width)
        self.width_mult = float(width)
        return self

    def active_out(self):
        return _active(self.c2_max, self.width_mult)

    def forward(self, x):
        cout = self.active_out()

        if self._is_depthwise():
            cout = x.shape[1]

        out_idx = torch.arange(cout, device=x.device)

        return self.forward_indexed(x, out_idx)
    
    def _is_depthwise(self):
        return (
            self.conv.groups == self.conv.in_channels
            and self.conv.groups == self.conv.out_channels
        )

    def forward_indexed(self, x, out_idx, in_idx=None):
        """
        Run this convolution using selected channels from the full-width weights.

        x is assumed to already contain the active channels in compact form.
        out_idx maps compact output channels -> full-width output channels.
        in_idx maps compact input channels -> full-width input channels.
        """

        expected = self.bn.active_features()

        if len(out_idx) != expected:
            raise RuntimeError(
                f"SlimConv selected {len(out_idx)} output channels, "
                f"but BN for width {self.width_mult} expects {expected}."
            )

        device = self.conv.weight.device

        out_idx = torch.as_tensor(out_idx, device=device, dtype=torch.long)

        if self.conv.groups == 1:
            weight = self.conv.weight.index_select(0, out_idx)

            if in_idx is None:
                # Ordinary prefix slimming.
                weight = weight[:, :x.shape[1], :, :]
            else:
                in_idx = torch.as_tensor(in_idx, device=device, dtype=torch.long)

                if len(in_idx) != x.shape[1]:
                    raise RuntimeError(
                        f"in_idx has {len(in_idx)} channels but input has {x.shape[1]}"
                    )

                weight = weight.index_select(1, in_idx)

            groups = 1

        elif self._is_depthwise():
            # For depthwise conv there is one kernel per input channel.
            if len(out_idx) != x.shape[1]:
                raise RuntimeError(
                    "Slim depthwise convolution requires one selected kernel "
                    "per active input channel."
                )

            weight = self.conv.weight.index_select(0, out_idx)
            groups = x.shape[1]

        else:
            raise RuntimeError(
                "SlimConv currently supports groups=1 or depthwise convolution only. "
                f"Got groups={self.conv.groups}."
            )

        y = F.conv2d(
            x,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=groups,
        )

        return self.act(self.bn(y))


class SlimPredConv(nn.Conv2d):
    def __init__(self, c1_max, c2):
        super().__init__(c1_max, c2, 1, bias=True)

    def set_width(self, width):
        pass

    def forward(self, x):
        if x.shape[1] == self.in_channels:
            return super().forward(x)

        weight = self.weight[:, :x.shape[1]]

        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class SlimBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3, 3), (3, 3)), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        k1 = k[0] if isinstance(k, tuple) and len(k) == 2 else 3
        k2 = k[1] if isinstance(k, tuple) and len(k) == 2 else 3

        self.cv1 = SlimConv(c1, c_, k1, 1)
        self.cv2 = SlimConv(c_, c2, k2, 1, g=g)
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
        input_idx = None

        # SlimConcat can pass:
        #   (compact_tensor, full_width_input_indices)
        if isinstance(x, tuple):
            x, input_idx = x

        active_c = _active(self.c, self.cv1.width_mult)

        cv1_idx = _group_prefix_indices(
            self.c,
            active_c,
            2,
            x.device,
        )

        x = self.cv1.forward_indexed(
            x,
            out_idx=cv1_idx,
            in_idx=input_idx,
        )

        y = [
            x[:, :active_c],
            x[:, active_c:],
        ]

        y.extend(m(y[-1]) for m in self.m)

        cat = torch.cat(y, dim=1)

        cv2_in_idx = _group_prefix_indices(
            self.c,
            active_c,
            len(y),
            x.device,
        )

        out_idx = torch.arange(
            self.cv2.active_out(),
            device=x.device,
        )

        return self.cv2.forward_indexed(
            cat,
            out_idx=out_idx,
            in_idx=cv2_in_idx,
        )

class SlimAttention(nn.Module):
    """
    Yu-style slimmable version of YOLO Attention.

    Keeps the number of attention heads fixed while reducing the
    dimensionality inside each head.

    Shared:
        qkv weights
        projection weights
        positional-encoding weights

    Width-specific:
        BatchNorm statistics through SlimConv/SwitchableBN2d
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_ratio: float = 0.5,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim_max = int(dim)
        self.num_heads = int(num_heads)
        self.attn_ratio = float(attn_ratio)

        self.head_dim_max = self.dim_max // self.num_heads
        self.key_dim_max = int(self.head_dim_max * self.attn_ratio)

        if self.key_dim_max < 1:
            raise ValueError("Attention key dimension must be >= 1")

        # Same maximum-width structure as stock YOLO Attention.
        qkv_channels = self.num_heads * (
            2 * self.key_dim_max + self.head_dim_max
        )

        self.qkv = SlimConv(
            self.dim_max,
            qkv_channels,
            1,
            act=False,
        )

        self.proj = SlimConv(
            self.dim_max,
            self.dim_max,
            1,
            act=False,
        )

        # Stock YOLO uses a depthwise 3x3 conv for positional encoding.
        self.pe = SlimConv(
            self.dim_max,
            self.dim_max,
            3,
            1,
            g=self.dim_max,
            act=False,
        )

        # Capture this model's supported widths.
        self.widths = tuple(self.qkv.bn.widths)
        
        self.width_mult = 1.0


    def set_width(self, width: float):
        _width_key(width, self.widths)

        self.qkv.set_width(width)
        self.proj.set_width(width)
        self.pe.set_width(width)

        self.width_mult = float(width)

        return self


    def _active_dims(self):
        dim = _active(self.dim_max, self.width_mult)

        if dim % self.num_heads != 0:
            raise RuntimeError(
                f"Active attention dim {dim} is not divisible by "
                f"{self.num_heads} heads."
            )

        head_dim = dim // self.num_heads
        key_dim = int(head_dim * self.attn_ratio)

        if key_dim < 1:
            raise RuntimeError("Active key dimension became < 1.")

        return dim, head_dim, key_dim


    def _qkv_indices(self, head_dim, key_dim, device):
        """
        Select the first active Q/K/V dimensions independently
        inside every full-width attention head.
        """

        indices = []

        full_per_head = (
            2 * self.key_dim_max + self.head_dim_max
        )

        for h in range(self.num_heads):
            base = h * full_per_head

            # Q
            indices.extend(
                range(
                    base,
                    base + key_dim,
                )
            )

            # K
            k_start = base + self.key_dim_max

            indices.extend(
                range(
                    k_start,
                    k_start + key_dim,
                )
            )

            # V
            v_start = base + 2 * self.key_dim_max

            indices.extend(
                range(
                    v_start,
                    v_start + head_dim,
                )
            )

        return torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )


    def _value_indices(self, head_dim, device):
        """
        Map the compact active V representation back to the
        corresponding channels of the full-width V representation.
        """

        indices = []

        for h in range(self.num_heads):
            start = h * self.head_dim_max

            indices.extend(
                range(
                    start,
                    start + head_dim,
                )
            )

        return torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )


    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        dim, head_dim, key_dim = self._active_dims()

        if C != dim:
            raise RuntimeError(
                f"SlimAttention expected {dim} active channels "
                f"at width {self.width_mult}, but received {C}."
            )

        # ---------------------------------------------------------
        # 1. QKV
        # ---------------------------------------------------------

        qkv_idx = self._qkv_indices(
            head_dim,
            key_dim,
            x.device,
        )

        qkv = self.qkv.forward_indexed(
            x,
            out_idx=qkv_idx,
        )

        per_head = 2 * key_dim + head_dim

        expected_qkv = self.num_heads * per_head

        if qkv.shape[1] != expected_qkv:
            raise RuntimeError(
                f"Expected {expected_qkv} QKV channels, "
                f"got {qkv.shape[1]}"
            )

        qkv = qkv.view(
            B,
            self.num_heads,
            per_head,
            N,
        )

        q, k, v = qkv.split(
            [key_dim, key_dim, head_dim],
            dim=2,
        )

        # ---------------------------------------------------------
        # 2. Attention
        # ---------------------------------------------------------

        scale = key_dim ** -0.5

        attn = (
            q.transpose(-2, -1) @ k
        ) * scale

        attn = attn.softmax(dim=-1)

        out = (
            v @ attn.transpose(-2, -1)
        )

        # Compact representation:
        #
        # head0 active values
        # head1 active values
        # ...
        out = out.reshape(
            B,
            dim,
            H,
            W,
        )

        # ---------------------------------------------------------
        # 3. Positional encoding
        # ---------------------------------------------------------

        value_idx = self._value_indices(
            head_dim,
            x.device,
        )

        pe = self.pe.forward_indexed(
            v.reshape(B, dim, H, W),
            out_idx=value_idx,
        )

        out = out + pe

        # ---------------------------------------------------------
        # 4. Projection
        # ---------------------------------------------------------
        #
        # Projection OUTPUT corresponds to the normal active prefix:
        #
        #     [0 ... dim_active]
        #
        # But its INPUT corresponds to the selected per-head
        # V dimensions.
        #

        output_idx = torch.arange(
            dim,
            device=x.device,
        )

        out = self.proj.forward_indexed(
            out,
            out_idx=output_idx,
            in_idx=value_idx,
        )

        return out

class SlimPSABlock(nn.Module):
    def __init__(
        self,
        c: int,
        attn_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
    ):
        super().__init__()

        self.attn = SlimAttention(
            c,
            attn_ratio=attn_ratio,
            num_heads=num_heads,
        )

        self.ffn = nn.Sequential(
            SlimConv(c, c * 2, 1),
            SlimConv(c * 2, c, 1, act=False),
        )

        self.add = shortcut


    def set_width(self, width: float):
        self.attn.set_width(width)

        for m in self.ffn:
            if hasattr(m, "set_width"):
                m.set_width(width)

        return self


    def forward(self, x):
        if self.add:
            x = x + self.attn(x)
        else:
            x = self.attn(x)

        if self.add:
            x = x + self.ffn(x)
        else:
            x = self.ffn(x)

        return x

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

        self.m = nn.ModuleList(
            nn.Sequential(
                SlimBottleneck(self.c, self.c, shortcut, g),
                SlimPSABlock(
                    self.c,
                    attn_ratio=0.5,
                    num_heads=max(self.c // 64, 1),
                ),
            )
            if attn
            else SlimC3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else SlimBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def set_width(self, width: float):
        self.width_mult = float(width)
        self.cv1.set_width(width)
        self.cv2.set_width(width)
        for m in self.m:
            if hasattr(m, "set_width"):
                m.set_width(width)
            else:
                for child in m:
                    if hasattr(child, "set_width"):
                        child.set_width(width)

    def forward(self, x):
        return super().forward(x)


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

        y.extend(
            self.m(y[-1])
            for _ in range(self.n)
        )

        cat = torch.cat(y, dim=1)

        # All pooled tensors have the same active channel count
        # and correspond to separate full-width groups.
        active_c = y[0].shape[1]

        in_idx = _group_prefix_indices(
            self.c,
            active_c,
            len(y),
            x.device,
        )

        out_idx = torch.arange(
            self.cv2.active_out(),
            device=x.device,
        )

        out = self.cv2.forward_indexed(
            cat,
            out_idx=out_idx,
            in_idx=in_idx,
        )

        return (
            out + x
            if self.add and out.shape == x.shape
            else out
        )


class SlimC2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()

        assert c1 == c2

        self.c = int(c1 * e)

        self.cv1 = SlimConv(c1, 2 * self.c, 1, 1)
        self.cv2 = SlimConv(2 * self.c, c1, 1)

        self.m = nn.Sequential(
            *(
                SlimPSABlock(
                    self.c,
                    attn_ratio=0.5,
                    num_heads=max(self.c // 64, 1),
                )
                for _ in range(n)
            )
        )

    def set_width(self, width):
        self.cv1.set_width(width)
        self.cv2.set_width(width)

        for m in self.m:
            m.set_width(width)

    def forward(self, x):
        active_c = _active(self.c, self.cv1.width_mult)

        split_idx = _group_prefix_indices(
            self.c,
            active_c,
            2,
            x.device,
        )

        x = self.cv1.forward_indexed(
            x,
            out_idx=split_idx,
        )

        a = x[:, :active_c]
        b = x[:, active_c:]

        b = self.m(b)

        x = torch.cat((a, b), dim=1)

        in_idx = _group_prefix_indices(
            self.c,
            active_c,
            2,
            x.device,
        )

        out_idx = torch.arange(
            self.cv2.active_out(),
            device=x.device,
        )

        return self.cv2.forward_indexed(
            x,
            out_idx=out_idx,
            in_idx=in_idx,
        )


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
            nn.Sequential(
                nn.Sequential(
                    SlimConv(x, x, 3, g=x),
                    SlimConv(x, c3, 1),
                ),
                nn.Sequential(
                    SlimConv(c3, c3, 3, g=c3),
                    SlimConv(c3, c3, 1),
                ),
                SlimPredConv(c3, self.nc),
            )
            for x in ch
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

class SlimC3k(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k=3,
    ):
        super().__init__()

        self.c = int(c2 * e)

        self.cv1 = SlimConv(c1, self.c, 1, 1)
        self.cv2 = SlimConv(c1, self.c, 1, 1)
        self.cv3 = SlimConv(2 * self.c, c2, 1, 1)

        self.m = nn.Sequential(
            *(
                SlimBottleneck(
                    self.c,
                    self.c,
                    shortcut,
                    g,
                    k=((k, k), (k, k)),
                    e=1.0,
                )
                for _ in range(n)
            )
        )

    def set_width(self, width: float):
        self.cv1.set_width(width)
        self.cv2.set_width(width)
        self.cv3.set_width(width)
        for m in self.m:
            if hasattr(m, "set_width"):
                m.set_width(width)

    def forward(self, x):
        a = self.m(self.cv1(x))
        b = self.cv2(x)

        if a.shape[1] != b.shape[1]:
            raise RuntimeError(
                f"SlimC3k branch mismatch: "
                f"{a.shape[1]} vs {b.shape[1]}"
            )

        active_c = a.shape[1]

        cat = torch.cat((a, b), dim=1)

        in_idx = _group_prefix_indices(
            self.c,
            active_c,
            2,
            x.device,
        )

        out_idx = torch.arange(
            self.cv3.active_out(),
            device=x.device,
        )

        return self.cv3.forward_indexed(
            cat,
            out_idx=out_idx,
            in_idx=in_idx,
        )

class SlimConcat(nn.Module):
    """
    Concatenate feature maps while preserving their full-width
    channel identities.

    At width 1.0 this behaves exactly like normal Concat.

    At slim widths it returns:
        (concatenated_tensor, full_width_channel_indices)

    The following SlimC2f/SlimC3k2 consumes that index mapping.
    """

    def __init__(
        self,
        dimension=1,
        ch=(),
    ):
        super().__init__()

        self.d = int(dimension)
        self.ch = tuple(int(c) for c in ch)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(
                "SlimConcat expects a list/tuple of tensors."
            )

        out = torch.cat(x, self.d)

        # We only need special channel bookkeeping
        # when concatenating channels.
        if self.d != 1:
            return out

        if not self.ch:
            raise RuntimeError(
                "SlimConcat requires the full-width channel "
                "counts for each input branch."
            )

        if len(x) != len(self.ch):
            raise RuntimeError(
                f"SlimConcat received {len(x)} tensors but "
                f"was configured for {len(self.ch)} branches."
            )

        active_sizes = [
            int(t.shape[1])
            for t in x
        ]

        for active_c, full_c in zip(
            active_sizes,
            self.ch,
        ):
            if active_c > full_c:
                raise RuntimeError(
                    f"SlimConcat active channels {active_c} "
                    f"exceed full channels {full_c}."
                )

        # At width 1.0 behave exactly like stock Concat.
        if all(
            active_c == full_c
            for active_c, full_c
            in zip(active_sizes, self.ch)
        ):
            return out

        in_idx = _concat_prefix_indices(
            self.ch,
            active_sizes,
            out.device,
        )

        return out, in_idx