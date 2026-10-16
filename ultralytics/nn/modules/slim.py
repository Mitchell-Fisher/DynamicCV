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

def _build_group_prefix_cache(
    module,
    cache_attr,
    buffer_prefix,
    group_max,
    num_groups,
    widths,
    device,
):
    """
    Precompute grouped-prefix channel-index tensors for all
    supported widths.

    These index mappings depend only on the architecture and width,
    not on the input image, so they can safely be reused.

    persistent=False keeps them out of state_dict().
    """

    existing = getattr(
        module,
        cache_attr,
        None,
    )

    # Cache already exists.
    if existing is not None:
        return existing

    names = {}

    for width in widths:

        key = _width_key(
            width,
            widths,
        )

        active_c = _active(
            group_max,
            width,
        )

        buffer_name = (
            f"_{buffer_prefix}_{key}"
        )

        # Support old serialized checkpoints that do not yet
        # contain these buffers.
        if buffer_name not in module._buffers:

            module.register_buffer(
                buffer_name,
                _group_prefix_indices(
                    group_max,
                    active_c,
                    num_groups,
                    device,
                ),
                persistent=False,
            )

        names[key] = buffer_name

    setattr(
        module,
        cache_attr,
        names,
    )

    return names


def _select_cached_group_prefix(
    module,
    cache_attr,
    buffer_prefix,
    active_attr,
    group_max,
    num_groups,
    width,
    widths,
    device,
):
    """
    Select the precomputed grouped-prefix index tensor for the
    requested width.
    """

    names = _build_group_prefix_cache(
        module=module,
        cache_attr=cache_attr,
        buffer_prefix=buffer_prefix,
        group_max=group_max,
        num_groups=num_groups,
        widths=widths,
        device=device,
    )

    key = _width_key(
        width,
        widths,
    )

    setattr(
        module,
        active_attr,
        names[key],
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

        self._use_cached_out_idx = True
        self._use_prefix_slicing = True
        self._use_full_width_fastpath = True

        self._build_out_idx_cache()

    def _build_out_idx_cache(self):
        """
        Build prefix output-channel index tensors once for all
        supported widths.

        This also provides backward compatibility with checkpoints
        created before these cached indices existed.
        """

        # If the cache already exists, do nothing.
        if hasattr(self, "_out_idx_names"):
            return

        self._out_idx_names = {}

        # Build the tensors on the same device as the convolution.
        device = self.conv.weight.device

        for width in self.bn.widths:
            key = _width_key(width, self.bn.widths)
            name = f"_cached_out_idx_{key}"

            self.register_buffer(
                name,
                torch.arange(
                    _active(self.c2_max, width),
                    device=device,
                    dtype=torch.long,
                ),
                persistent=False,
            )

            self._out_idx_names[key] = name

        # Match the cache selection to whatever width the loaded
        # module currently has.
        key = _width_key(
            self.width_mult,
            self.bn.widths,
        )

        self._active_out_idx_name = self._out_idx_names[key]

    def set_width(self, width: float):
        # ----------------------------------------------------------
        # Important for old checkpoints:
        #
        # torch.load() restores the saved SlimConv object without
        # rerunning this class's new __init__(). Therefore an older
        # checkpoint will not yet contain _out_idx_names.
        # ----------------------------------------------------------
        if not hasattr(self, "_out_idx_names"):
            self._build_out_idx_cache()

        key = _width_key(
            width,
            self.bn.widths,
        )

        self.bn.set_width(width)
        self.width_mult = float(width)

        self._active_out_idx_name = self._out_idx_names[key]

        return self

    def active_out(self):
        return _active(
            self.c2_max,
            self.width_mult,
        )

    def _forward_prefix(self, x):
        """
        Fast path for ordinary SlimConv.forward() calls.

        Both input and output channels are known to be compact
        contiguous prefixes, so weight slicing can be used instead
        of index_select.

        This method is NOT used by forward_indexed(), because those
        callers may use grouped or otherwise non-contiguous mappings.
        """

        expected = self.bn.active_features()

        # ----------------------------------------------------------
        # Standard convolution: groups == 1
        # ----------------------------------------------------------

        if self.conv.groups == 1:

            cout = self.active_out()
            cin = x.shape[1]

            if cout != expected:
                raise RuntimeError(
                    f"SlimConv prefix path selected {cout} output channels, "
                    f"but BN for width {self.width_mult} expects {expected}."
                )

            if cin > self.conv.in_channels:
                raise RuntimeError(
                    f"SlimConv received {cin} input channels, "
                    f"but full-width convolution has only "
                    f"{self.conv.in_channels}."
                )

            # ------------------------------------------------------
            # Optimization 2A-1
            #
            # OLD:
            #
            # weight = self.conv.weight.index_select(0, out_idx)
            # weight = weight[:, :cin, :, :]
            #
            # NEW:
            #
            # Direct contiguous-prefix view into the original weight.
            # ------------------------------------------------------

            weight = self.conv.weight[
                :cout,
                :cin,
                :,
                :,
            ]

            groups = 1

        # ----------------------------------------------------------
        # Depthwise convolution
        # ----------------------------------------------------------

        elif self._is_depthwise():

            cout = x.shape[1]

            if cout != expected:
                raise RuntimeError(
                    f"Slim depthwise convolution received {cout} "
                    f"active channels, but BN for width "
                    f"{self.width_mult} expects {expected}."
                )

            if cout > self.conv.out_channels:
                raise RuntimeError(
                    f"Slim depthwise convolution requested {cout} kernels, "
                    f"but only {self.conv.out_channels} exist."
                )

            # One kernel per active input channel.
            #
            # OLD:
            # weight = self.conv.weight.index_select(0, out_idx)
            #
            # NEW:
            # contiguous prefix slice.
            weight = self.conv.weight[
                :cout,
                :,
                :,
                :,
            ]

            groups = cout

        else:

            raise RuntimeError(
                "SlimConv currently supports groups=1 or "
                "depthwise convolution only. "
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

        return self.act(
            self.bn(y)
        )

    def forward_materialized(
            self,
            x,
            weight,
    ):
        """
        Run SlimConv using an already-materialized compact convolution
        weight.

        Optimization 2B-2 uses this for SlimC3k2 inference after the
        required non-contiguous channel selections have been performed
        once and cached.

        This path is inference-oriented. The supplied weight must already
        correspond exactly to the compact input/output channel layout.
        """

        if self.conv.groups != 1:
            raise RuntimeError(
                "forward_materialized() currently supports "
                "groups=1 only."
            )

        expected = self.bn.active_features()

        if weight.shape[0] != expected:
            raise RuntimeError(
                f"Materialized SlimConv weight has "
                f"{weight.shape[0]} output channels, "
                f"but BN expects {expected}."
            )

        if weight.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"Materialized SlimConv weight has "
                f"{weight.shape[1]} input channels, "
                f"but input tensor has {x.shape[1]}."
            )

        y = F.conv2d(
            x,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=1,
        )

        return self.act(
            self.bn(y)
        )

    def forward_materialized_depthwise(
            self,
            x,
            weight,
    ):
        """
        Run a depthwise SlimConv using an already-materialized compact
        kernel tensor.

        Used by Optimization 2B-5A for SlimAttention positional
        encoding.

        The supplied weight must already contain one kernel for each
        compact input channel, in exactly the same channel order as x.
        """

        if not self._is_depthwise():
            raise RuntimeError(
                "forward_materialized_depthwise() requires a "
                "depthwise SlimConv."
            )

        expected = self.bn.active_features()

        if weight.shape[0] != expected:
            raise RuntimeError(
                f"Materialized depthwise weight has "
                f"{weight.shape[0]} output channels, "
                f"but BN expects {expected}."
            )

        if weight.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Materialized depthwise weight has "
                f"{weight.shape[0]} kernels, but input tensor "
                f"has {x.shape[1]} channels."
            )

        if weight.shape[1] != 1:
            raise RuntimeError(
                "Materialized depthwise convolution weight must "
                "have shape [C, 1, kH, kW]."
            )

        y = F.conv2d(
            x,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=x.shape[1],
        )

        return self.act(
            self.bn(y)
        )

    def forward(self, x):

        # ==========================================================
        # OPTIMIZATION 2A-1
        #
        # Ordinary SlimConv.forward() always represents compact
        # contiguous input/output prefixes.
        #
        # Therefore we do not need index_select at all.
        # ==========================================================

        use_prefix_slicing = getattr(
            self,
            "_use_prefix_slicing",
            True,
        )

        if use_prefix_slicing:
            return self._forward_prefix(x)

        # ==========================================================
        # BASELINE PATH
        #
        # Keep the current 1A implementation intact so we can perform
        # a controlled A/B comparison.
        # ==========================================================

        use_cache = getattr(
            self,
            "_use_cached_out_idx",
            True,
        )

        # ----------------------------------------------------------
        # Original pre-1A mode.
        # ----------------------------------------------------------

        if not use_cache:

            cout = self.active_out()

            if self._is_depthwise():
                cout = x.shape[1]

            out_idx = torch.arange(
                cout,
                device=x.device,
                dtype=torch.long,
            )

            return self.forward_indexed(
                x,
                out_idx,
            )

        # ----------------------------------------------------------
        # Optimization 1A mode:
        # cached output-prefix index tensor.
        # ----------------------------------------------------------

        if not hasattr(
                self,
                "_out_idx_names",
        ):
            self._build_out_idx_cache()

        out_idx = getattr(
            self,
            self._active_out_idx_name,
        )

        if (
                self._is_depthwise()
                and out_idx.numel() != x.shape[1]
        ):
            out_idx = torch.arange(
                x.shape[1],
                device=x.device,
                dtype=torch.long,
            )

        return self.forward_indexed(
            x,
            out_idx,
        )
    
    def _is_depthwise(self):
        return (
            self.conv.groups == self.conv.in_channels
            and self.conv.groups == self.conv.out_channels
        )

    def forward_indexed(self, x, out_idx, in_idx=None):
        """
        Run this convolution using selected channels from the
        full-width weights.

        Optimization 2B-1:
            At width 1.0, all special index mappings collapse to identity
            mappings, so no weight selection is required.
        """

        # ==========================================================
        # OPTIMIZATION 2B-1: FULL-WIDTH IDENTITY FAST PATH
        # ==========================================================

        use_full_width_fastpath = getattr(
            self,
            "_use_full_width_fastpath",
            True,
        )

        if (
                use_full_width_fastpath
                and abs(self.width_mult - 1.0) < 1e-6
        ):
            # At full width the compact tensor must contain every
            # original input channel.
            if x.shape[1] != self.conv.in_channels:
                raise RuntimeError(
                    f"Full-width SlimConv fast path received "
                    f"{x.shape[1]} input channels, but convolution "
                    f"expects {self.conv.in_channels}."
                )

            # Full-width BN should likewise expect the complete
            # convolution output.
            expected = self.bn.active_features()

            if expected != self.conv.out_channels:
                raise RuntimeError(
                    f"Full-width SlimConv fast path has "
                    f"{self.conv.out_channels} convolution outputs, "
                    f"but BN expects {expected}."
                )

            # ------------------------------------------------------
            # No index_select.
            # No slicing.
            # No temporary weight tensor.
            #
            # Use the original full-width convolution weights
            # directly.
            # ------------------------------------------------------

            y = F.conv2d(
                x,
                self.conv.weight,
                bias=None,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

            return self.act(
                self.bn(y)
            )

        # ==========================================================
        # EXISTING SLIMMABLE PATH
        #
        # Everything below remains unchanged.
        # ==========================================================

        expected = self.bn.active_features()

        if len(out_idx) != expected:
            raise RuntimeError(
                f"SlimConv selected {len(out_idx)} output channels, "
                f"but BN for width {self.width_mult} expects {expected}."
            )

        device = self.conv.weight.device

        out_idx = torch.as_tensor(
            out_idx,
            device=device,
            dtype=torch.long,
        )

        if self.conv.groups == 1:

            weight = self.conv.weight.index_select(
                0,
                out_idx,
            )

            if in_idx is None:

                # Ordinary prefix slimming.
                weight = weight[
                    :,
                    :x.shape[1],
                    :,
                    :,
                ]

            else:

                in_idx = torch.as_tensor(
                    in_idx,
                    device=device,
                    dtype=torch.long,
                )

                if len(in_idx) != x.shape[1]:
                    raise RuntimeError(
                        f"in_idx has {len(in_idx)} channels "
                        f"but input has {x.shape[1]}"
                    )

                weight = weight.index_select(
                    1,
                    in_idx,
                )

            groups = 1

        elif self._is_depthwise():

            # For depthwise conv there is one kernel per input channel.
            if len(out_idx) != x.shape[1]:
                raise RuntimeError(
                    "Slim depthwise convolution requires one selected "
                    "kernel per active input channel."
                )

            weight = self.conv.weight.index_select(
                0,
                out_idx,
            )

            groups = x.shape[1]

        else:

            raise RuntimeError(
                "SlimConv currently supports groups=1 or "
                "depthwise convolution only. "
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

        return self.act(
            self.bn(y)
        )

    def set_cached_indices(self, enabled: bool):
        """
        Enable or disable cached output-prefix indices.

        enabled=True:
            Reuse precomputed prefix indices.

        enabled=False:
            Reproduce the original behavior by constructing
            torch.arange() every forward.
        """
        self._use_cached_out_idx = bool(enabled)
        return self

    def set_prefix_slicing(
            self,
            enabled: bool,
    ):
        """
        Enable/disable Optimization 2A-1.

        enabled=True:
            Ordinary SlimConv.forward() uses contiguous weight slices.

        enabled=False:
            Ordinary SlimConv.forward() uses the existing
            cached-index + forward_indexed() implementation.
        """

        self._use_prefix_slicing = bool(
            enabled
        )

        return self

    def set_full_width_fastpath(
            self,
            enabled: bool,
    ):
        """
        Enable/disable Optimization 2B-1.

        enabled=True:
            At width 1.0, forward_indexed() bypasses all channel
            selection and uses the original full convolution weight
            directly.

        enabled=False:
            Reproduce the existing indexed full-width behavior.

        Narrow widths are unaffected in either mode.
        """

        self._use_full_width_fastpath = bool(
            enabled
        )

        return self

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

        # Old serialized checkpoints won't have these new cache
        # attributes because __init__ is not rerun during torch.load().
        if not hasattr(
                self,
                "_cv1_group_idx_names",
        ):
            self._build_group_idx_cache()

        self._select_group_idx_cache(
            width
        )

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

        use_group_cache = getattr(
            self,
            "_use_cached_group_idx",
            True,
        )

        if use_group_cache:

            if not hasattr(
                    self,
                    "_active_cv1_group_idx_name",
            ):
                self._build_group_idx_cache()
                self._select_group_idx_cache(
                    self.cv1.width_mult
                )

            cv1_idx = getattr(
                self,
                self._active_cv1_group_idx_name,
            )

        else:
            # Original implementation for A/B testing.
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

        if use_group_cache:

            cv2_in_idx = getattr(
                self,
                self._active_cv2_group_idx_name,
            )

        else:
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

    def _build_group_idx_cache(self):
        """
        Build the two grouped-prefix mappings used by C2f:

        cv1:
            2 logical groups

        cv2 input:
            2 + number of bottleneck outputs
        """

        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _build_group_prefix_cache(
            module=self,
            cache_attr="_cv1_group_idx_names",
            buffer_prefix="cached_c2f_cv1_group_idx",
            group_max=self.c,
            num_groups=2,
            widths=widths,
            device=device,
        )

        _build_group_prefix_cache(
            module=self,
            cache_attr="_cv2_group_idx_names",
            buffer_prefix="cached_c2f_cv2_group_idx",
            group_max=self.c,
            num_groups=2 + len(self.m),
            widths=widths,
            device=device,
        )

    def _select_group_idx_cache(self, width):
        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _select_cached_group_prefix(
            module=self,
            cache_attr="_cv1_group_idx_names",
            buffer_prefix="cached_c2f_cv1_group_idx",
            active_attr="_active_cv1_group_idx_name",
            group_max=self.c,
            num_groups=2,
            width=width,
            widths=widths,
            device=device,
        )

        _select_cached_group_prefix(
            module=self,
            cache_attr="_cv2_group_idx_names",
            buffer_prefix="cached_c2f_cv2_group_idx",
            active_attr="_active_cv2_group_idx_name",
            group_max=self.c,
            num_groups=2 + len(self.m),
            width=width,
            widths=widths,
            device=device,
        )

    def set_cached_group_indices(self, enabled: bool):
        self._use_cached_group_idx = bool(
            enabled
        )
        return self

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
        # Cache the complete narrow-width QKV, projection, and
        # positional-encoding convolution weights.
        self._use_compact_attention_weight_cache = True
        self._compact_attention_weight_cache_meta = {}


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

        dim, head_dim, key_dim = (
            self._active_dims()
        )

        if C != dim:
            raise RuntimeError(
                f"SlimAttention expected {dim} active channels "
                f"at width {self.width_mult}, but received {C}."
            )

        # ==========================================================
        # OPTIMIZATION 2B-5A
        # ==========================================================

        use_compact_cache = getattr(
            self,
            "_use_compact_attention_weight_cache",
            True,
        )

        use_materialized_weights = (
                use_compact_cache
                and not self.training
                and abs(
            self.width_mult - 1.0
        ) >= 1e-6
        )

        if use_materialized_weights:

            (
                qkv_weight,
                pe_weight,
                proj_weight,
            ) = self._get_compact_attention_weights()

            # ------------------------------------------------------
            # QKV
            # ------------------------------------------------------

            qkv = self.qkv.forward_materialized(
                x,
                qkv_weight,
            )

        else:

            # ------------------------------------------------------
            # ORIGINAL QKV PATH
            # ------------------------------------------------------

            qkv_idx = self._qkv_indices(
                head_dim,
                key_dim,
                x.device,
            )

            qkv = self.qkv.forward_indexed(
                x,
                out_idx=qkv_idx,
            )

        # ==========================================================
        # Q / K / V processing - unchanged
        # ==========================================================

        per_head = (
                2 * key_dim
                + head_dim
        )

        expected_qkv = (
                self.num_heads
                * per_head
        )

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
            [
                key_dim,
                key_dim,
                head_dim,
            ],
            dim=2,
        )

        scale = (
                key_dim ** -0.5
        )

        attn = (
                       q.transpose(-2, -1)
                       @ k
               ) * scale

        attn = attn.softmax(
            dim=-1
        )

        out = (
                v
                @ attn.transpose(
            -2,
            -1,
        )
        )

        out = out.reshape(
            B,
            dim,
            H,
            W,
        )

        # ==========================================================
        # POSITIONAL ENCODING
        # ==========================================================

        compact_v = v.reshape(
            B,
            dim,
            H,
            W,
        )

        if use_materialized_weights:

            pe = (
                self.pe.forward_materialized_depthwise(
                    compact_v,
                    pe_weight,
                )
            )

        else:

            value_idx = self._value_indices(
                head_dim,
                x.device,
            )

            pe = self.pe.forward_indexed(
                compact_v,
                out_idx=value_idx,
            )

        out = out + pe

        # ==========================================================
        # PROJECTION
        # ==========================================================

        if use_materialized_weights:

            out = (
                self.proj.forward_materialized(
                    out,
                    proj_weight,
                )
            )

        else:

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

    def _init_compact_attention_weight_cache(self):
        """
        Lazily initialize 2B-5A cache metadata.

        Required for older pickled checkpoints whose __init__()
        is not rerun during torch.load().
        """

        if not hasattr(
                self,
                "_compact_attention_weight_cache_meta",
        ):
            self._compact_attention_weight_cache_meta = {}

    def _build_compact_attention_weights(self):
        """
        Materialize the three special SlimAttention weights needed
        by the current narrow width:

            qkv  - non-contiguous output mapping
            pe   - non-contiguous depthwise kernel mapping
            proj - prefix outputs + non-contiguous inputs

        These mappings depend only on width and architecture, so the
        resulting compact weights can be reused across inference calls.
        """

        self._init_compact_attention_weight_cache()

        width = float(
            self.width_mult
        )

        if abs(width - 1.0) < 1e-6:
            raise RuntimeError(
                "Compact SlimAttention weights should not be built "
                "at width 1.0. Optimization 2B-1 handles full width."
            )

        key = _width_key(
            width,
            self.widths,
        )

        dim, head_dim, key_dim = (
            self._active_dims()
        )

        device = (
            self.qkv.conv.weight.device
        )

        # ==========================================================
        # Build the same mappings used by the original forward path.
        # ==========================================================

        qkv_idx = self._qkv_indices(
            head_dim,
            key_dim,
            device,
        )

        value_idx = self._value_indices(
            head_dim,
            device,
        )

        # ==========================================================
        # QKV
        #
        # Original:
        #   non-contiguous output gather
        #   ordinary compact input prefix
        # ==========================================================

        qkv_weight = (
            self.qkv.conv.weight
            .index_select(
                0,
                qkv_idx,
            )
        )

        qkv_weight = qkv_weight[
            :,
            :dim,
            :,
            :,
        ]

        # ==========================================================
        # Positional encoding
        #
        # Depthwise: one selected kernel for every selected V channel.
        # ==========================================================

        pe_weight = (
            self.pe.conv.weight
            .index_select(
                0,
                value_idx,
            )
        )

        # ==========================================================
        # Projection
        #
        # Output is normal active prefix.
        # Input corresponds to the per-head active V mapping.
        # ==========================================================

        proj_weight = self.proj.conv.weight[
            :dim,
            :,
            :,
            :,
        ]

        proj_weight = (
            proj_weight.index_select(
                1,
                value_idx,
            )
        )

        # ==========================================================
        # Make independent contiguous inference buffers.
        # ==========================================================

        qkv_weight = (
            qkv_weight
            .detach()
            .contiguous()
            .clone()
        )

        pe_weight = (
            pe_weight
            .detach()
            .contiguous()
            .clone()
        )

        proj_weight = (
            proj_weight
            .detach()
            .contiguous()
            .clone()
        )

        qkv_name = (
            f"_cached_attention_qkv_weight_{key}"
        )

        pe_name = (
            f"_cached_attention_pe_weight_{key}"
        )

        proj_name = (
            f"_cached_attention_proj_weight_{key}"
        )

        buffers = (
            (qkv_name, qkv_weight),
            (pe_name, pe_weight),
            (proj_name, proj_weight),
        )

        for name, weight in buffers:

            if name in self._buffers:
                self._buffers[name] = weight

            else:
                self.register_buffer(
                    name,
                    weight,
                    persistent=False,
                )

        self._compact_attention_weight_cache_meta[
            key
        ] = {
            "qkv_name": qkv_name,
            "pe_name": pe_name,
            "proj_name": proj_name,

            "qkv_version":
                self.qkv.conv.weight._version,

            "pe_version":
                self.pe.conv.weight._version,

            "proj_version":
                self.proj.conv.weight._version,

            "dim":
                int(dim),

            "head_dim":
                int(head_dim),

            "key_dim":
                int(key_dim),
        }

        return (
            getattr(
                self,
                qkv_name,
            ),
            getattr(
                self,
                pe_name,
            ),
            getattr(
                self,
                proj_name,
            ),
        )

    def _get_compact_attention_weights(self):
        """
        Return valid cached weights for the active width.

        Rebuild if any underlying full-width convolution parameter
        changed.
        """

        self._init_compact_attention_weight_cache()

        width = float(
            self.width_mult
        )

        key = _width_key(
            width,
            self.widths,
        )

        dim, head_dim, key_dim = (
            self._active_dims()
        )

        meta = (
            self._compact_attention_weight_cache_meta.get(
                key
            )
        )

        cache_valid = (
                meta is not None

                and meta["qkv_name"] in self._buffers
                and meta["pe_name"] in self._buffers
                and meta["proj_name"] in self._buffers

                and meta["qkv_version"]
                == self.qkv.conv.weight._version

                and meta["pe_version"]
                == self.pe.conv.weight._version

                and meta["proj_version"]
                == self.proj.conv.weight._version

                and meta["dim"]
                == int(dim)

                and meta["head_dim"]
                == int(head_dim)

                and meta["key_dim"]
                == int(key_dim)
        )

        if not cache_valid:
            return (
                self._build_compact_attention_weights()
            )

        return (
            getattr(
                self,
                meta["qkv_name"],
            ),
            getattr(
                self,
                meta["pe_name"],
            ),
            getattr(
                self,
                meta["proj_name"],
            ),
        )

    def set_compact_attention_weight_cache(
            self,
            enabled: bool,
    ):
        """
        Enable/disable Optimization 2B-5A.

        ON:
            narrow-width eval inference reuses materialized
            qkv / pe / proj weights.

        OFF:
            reproduce the existing SlimAttention indexed path.

        Training and width 1.0 always follow the original path.
        """

        self._use_compact_attention_weight_cache = bool(
            enabled
        )

        return self

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

        # Optimization 2B-2:
        #
        # At narrow widths, cache the fully materialized compact cv1/cv2
        # weights used by SlimC3k2 so that repeated inference does not
        # perform the same index_select operations every image.
        self._use_compact_weight_cache = True

        # Metadata only. Actual tensor buffers are built lazily because
        # old pickled checkpoints do not rerun __init__().
        self._compact_weight_cache_meta = {}
        self._compact_weight_buffer_names = set()

    def set_width(self, width: float):

        self.width_mult = float(width)

        # ----------------------------------------------------------
        # Optimization 1B-1:
        #
        # SlimC3k2 inherits SlimC2f.forward(), so it must also
        # update the grouped-prefix cache used by that forward().
        #
        # Older serialized checkpoints will not contain the new
        # cache attributes, so build them lazily if needed.
        # ----------------------------------------------------------

        if not hasattr(
                self,
                "_cv1_group_idx_names",
        ):
            self._build_group_idx_cache()

        self._select_group_idx_cache(
            width
        )

        # ----------------------------------------------------------
        # Existing width propagation
        # ----------------------------------------------------------

        self.cv1.set_width(width)
        self.cv2.set_width(width)

        for m in self.m:

            if hasattr(
                    m,
                    "set_width",
            ):
                m.set_width(width)

            else:

                # Handles nn.Sequential containers.
                for child in m:

                    if hasattr(
                            child,
                            "set_width",
                    ):
                        child.set_width(width)

        return self

    def forward(self, x):

        use_compact_cache = getattr(
            self,
            "_use_compact_weight_cache",
            True,
        )

        # ==========================================================
        # ORIGINAL PATH
        #
        # Keep training untouched.
        #
        # Keep full width on the existing SlimC2f path so that
        # Optimization 2B-1 continues to handle width 1.0.
        # ==========================================================

        if (
                not use_compact_cache
                or self.training
                or abs(self.cv1.width_mult - 1.0) < 1e-6
        ):
            return super().forward(x)

        # ==========================================================
        # OPTIMIZATION 2B-2
        # ==========================================================

        input_idx = None

        # SlimConcat may supply:
        #
        #   (compact_tensor, full_width_input_indices)
        #
        if isinstance(
                x,
                tuple,
        ):
            x, input_idx = x

        active_c = _active(
            self.c,
            self.cv1.width_mult,
        )

        # ----------------------------------------------------------
        # Existing 1B-1 grouped mappings.
        # ----------------------------------------------------------

        use_group_cache = getattr(
            self,
            "_use_cached_group_idx",
            True,
        )

        if use_group_cache:

            if not hasattr(
                    self,
                    "_active_cv1_group_idx_name",
            ):
                self._build_group_idx_cache()

                self._select_group_idx_cache(
                    self.cv1.width_mult
                )

            cv1_idx = getattr(
                self,
                self._active_cv1_group_idx_name,
            )

            cv2_in_idx = getattr(
                self,
                self._active_cv2_group_idx_name,
            )

        else:

            cv1_idx = _group_prefix_indices(
                self.c,
                active_c,
                2,
                x.device,
            )

            cv2_in_idx = _group_prefix_indices(
                self.c,
                active_c,
                2 + len(self.m),
                x.device,
            )

        # ----------------------------------------------------------
        # Obtain the two already-compacted convolution weights.
        #
        # First inference at this width:
        #     build + cache
        #
        # Later inference:
        #     direct reuse
        # ----------------------------------------------------------

        cv1_weight, cv2_weight = (
            self._get_compact_weights(
                x=x,
                input_idx=input_idx,
                cv1_idx=cv1_idx,
                cv2_in_idx=cv2_in_idx,
            )
        )

        # ----------------------------------------------------------
        # CV1 with no runtime weight selection.
        # ----------------------------------------------------------

        x = self.cv1.forward_materialized(
            x,
            cv1_weight,
        )

        y = [
            x[
                :,
                :active_c,
            ],
            x[
                :,
                active_c:,
            ],
        ]

        y.extend(
            m(y[-1])
            for m in self.m
        )

        cat = torch.cat(
            y,
            dim=1,
        )

        # ----------------------------------------------------------
        # CV2 with no runtime output or input weight selection.
        # ----------------------------------------------------------

        return self.cv2.forward_materialized(
            cat,
            cv2_weight,
        )

    def _init_compact_weight_cache(self):
        """
        Lazily initialize Optimization 2B-2 metadata.

        Required because older serialized YOLO checkpoints restore
        SlimC3k2 objects without rerunning the current __init__().
        """

        if not hasattr(
                self,
                "_compact_weight_cache_meta",
        ):
            self._compact_weight_cache_meta = {}

        if not hasattr(
                self,
                "_compact_weight_buffer_names",
        ):
            self._compact_weight_buffer_names = set()

    def _store_compact_weight(
            self,
            name,
            weight,
    ):
        """
        Store a compact inference weight as a nonpersistent buffer.

        Nonpersistent:
            - follows .cpu(), .cuda(), .to(), .half(), etc.
            - is not written into state_dict()
        """

        self._init_compact_weight_cache()

        weight = (
            weight
            .detach()
            .contiguous()
            .clone()
        )

        if name in self._buffers:
            self._buffers[name] = weight
        else:
            self.register_buffer(
                name,
                weight,
                persistent=False,
            )

        self._compact_weight_buffer_names.add(
            name
        )

    def _build_compact_weights(
            self,
            x,
            input_idx,
            cv1_idx,
            cv2_in_idx,
    ):
        """
        Build the complete compact cv1 and cv2 weights for the currently
        active narrow width.

        The exact index tensors already produced by the existing
        architecture are used, so the cached weights represent exactly
        the same channels as the normal forward_indexed() path.
        """

        self._init_compact_weight_cache()

        width = float(
            self.cv1.width_mult
        )

        if abs(width - 1.0) < 1e-6:
            raise RuntimeError(
                "2B-2 compact weights should not be built "
                "for width 1.0. Optimization 2B-1 handles "
                "the full-width case."
            )

        widths = tuple(
            self.cv1.bn.widths
        )

        key = _width_key(
            width,
            widths,
        )

        cv1_name = (
            f"_cached_c3k2_cv1_weight_{key}"
        )

        cv2_name = (
            f"_cached_c3k2_cv2_weight_{key}"
        )

        # ==========================================================
        # CV1
        #
        # Existing runtime path:
        #
        #   weight.index_select(0, cv1_idx)
        #
        # followed by either:
        #
        #   [:, :x.shape[1]]
        #
        # or:
        #
        #   index_select(1, input_idx)
        #
        # ==========================================================

        cv1_idx = torch.as_tensor(
            cv1_idx,
            device=self.cv1.conv.weight.device,
            dtype=torch.long,
        )

        cv1_weight = (
            self.cv1.conv.weight.index_select(
                0,
                cv1_idx,
            )
        )

        if input_idx is None:

            cv1_weight = cv1_weight[
                :,
                :x.shape[1],
                :,
                :,
            ]

        else:

            input_idx = torch.as_tensor(
                input_idx,
                device=self.cv1.conv.weight.device,
                dtype=torch.long,
            )

            if len(input_idx) != x.shape[1]:
                raise RuntimeError(
                    f"SlimC3k2 input_idx has "
                    f"{len(input_idx)} channels but "
                    f"input has {x.shape[1]}."
                )

            cv1_weight = (
                cv1_weight.index_select(
                    1,
                    input_idx,
                )
            )

        # ==========================================================
        # CV2
        #
        # Existing path selects a normal active output prefix, then
        # selects the non-contiguous grouped input mapping.
        #
        # Because we are materializing the complete compact weight
        # once, we can take the output prefix directly here and perform
        # only the required input gather during cache construction.
        # ==========================================================

        cout = self.cv2.active_out()

        cv2_in_idx = torch.as_tensor(
            cv2_in_idx,
            device=self.cv2.conv.weight.device,
            dtype=torch.long,
        )

        cv2_weight = self.cv2.conv.weight[
            :cout,
            :,
            :,
            :,
        ]

        cv2_weight = (
            cv2_weight.index_select(
                1,
                cv2_in_idx,
            )
        )

        # ----------------------------------------------------------
        # Store final compact tensors.
        # ----------------------------------------------------------

        self._store_compact_weight(
            cv1_name,
            cv1_weight,
        )

        self._store_compact_weight(
            cv2_name,
            cv2_weight,
        )

        # ----------------------------------------------------------
        # Record the parameter versions that produced the cache.
        #
        # PyTorch Parameter._version changes when the underlying
        # parameter is modified in-place, such as during training or
        # state loading. This lets us rebuild stale inference caches.
        # ----------------------------------------------------------

        self._compact_weight_cache_meta[
            key
        ] = {
            "cv1_name": cv1_name,
            "cv2_name": cv2_name,

            "cv1_version": (
                self.cv1.conv.weight._version
            ),

            "cv2_version": (
                self.cv2.conv.weight._version
            ),

            "cv1_in_channels": int(
                x.shape[1]
            ),

            "cv2_in_channels": int(
                len(cv2_in_idx)
            ),
        }

        return (
            getattr(
                self,
                cv1_name,
            ),
            getattr(
                self,
                cv2_name,
            ),
        )

    def _get_compact_weights(
            self,
            x,
            input_idx,
            cv1_idx,
            cv2_in_idx,
    ):
        """
        Return valid cached compact weights for the active width,
        rebuilding them if necessary.
        """

        self._init_compact_weight_cache()

        width = float(
            self.cv1.width_mult
        )

        widths = tuple(
            self.cv1.bn.widths
        )

        key = _width_key(
            width,
            widths,
        )

        meta = (
            self._compact_weight_cache_meta.get(
                key
            )
        )

        cache_valid = (
                meta is not None
                and meta["cv1_name"] in self._buffers
                and meta["cv2_name"] in self._buffers
                and meta["cv1_version"]
                == self.cv1.conv.weight._version
                and meta["cv2_version"]
                == self.cv2.conv.weight._version
                and meta["cv1_in_channels"]
                == int(x.shape[1])
                and meta["cv2_in_channels"]
                == int(len(cv2_in_idx))
        )

        if not cache_valid:
            return self._build_compact_weights(
                x=x,
                input_idx=input_idx,
                cv1_idx=cv1_idx,
                cv2_in_idx=cv2_in_idx,
            )

        return (
            getattr(
                self,
                meta["cv1_name"],
            ),
            getattr(
                self,
                meta["cv2_name"],
            ),
        )

    def set_compact_weight_cache(
            self,
            enabled: bool,
    ):
        """
        Enable/disable Optimization 2B-2.

        enabled=True:
            At narrow widths in eval mode, SlimC3k2 uses
            pre-materialized compact cv1/cv2 weights.

        enabled=False:
            SlimC3k2 follows the original inherited SlimC2f
            forward_indexed() path.

        Width 1.0 continues to use Optimization 2B-1.
        Training always uses the original parameter path.
        """

        self._use_compact_weight_cache = bool(
            enabled
        )

        return self

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
        if not hasattr(
                self,
                "_group_idx_names",
        ):
            self._build_group_idx_cache()

        self._select_group_idx_cache(
            width
        )

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

        use_group_cache = getattr(
            self,
            "_use_cached_group_idx",
            True,
        )

        if use_group_cache:

            if not hasattr(
                    self,
                    "_active_group_idx_name",
            ):
                self._build_group_idx_cache()
                self._select_group_idx_cache(
                    self.cv1.width_mult
                )

            in_idx = getattr(
                self,
                self._active_group_idx_name,
            )

        else:

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

    def _build_group_idx_cache(self):

        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _build_group_prefix_cache(
            module=self,
            cache_attr="_group_idx_names",
            buffer_prefix="cached_sppf_group_idx",
            group_max=self.c,
            num_groups=self.n + 1,
            widths=widths,
            device=device,
        )


    def _select_group_idx_cache(self, width):

        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _select_cached_group_prefix(
            module=self,
            cache_attr="_group_idx_names",
            buffer_prefix="cached_sppf_group_idx",
            active_attr="_active_group_idx_name",
            group_max=self.c,
            num_groups=self.n + 1,
            width=width,
            widths=widths,
            device=device,
        )


    def set_cached_group_indices(self, enabled: bool):
        self._use_cached_group_idx = bool(
            enabled
        )
        return self

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

        if not hasattr(
                self,
                "_group_idx_names",
        ):
            self._build_group_idx_cache()

        self._select_group_idx_cache(
            width
        )

        self.cv1.set_width(width)
        self.cv2.set_width(width)

        for m in self.m:
            m.set_width(width)

    def forward(self, x):
        active_c = _active(self.c, self.cv1.width_mult)

        use_group_cache = getattr(
            self,
            "_use_cached_group_idx",
            True,
        )

        if use_group_cache:

            if not hasattr(
                    self,
                    "_active_group_idx_name",
            ):
                self._build_group_idx_cache()
                self._select_group_idx_cache(
                    self.cv1.width_mult
                )

            group_idx = getattr(
                self,
                self._active_group_idx_name,
            )

        else:

            group_idx = _group_prefix_indices(
                self.c,
                active_c,
                2,
                x.device,
            )

        split_idx = group_idx

        x = self.cv1.forward_indexed(
            x,
            out_idx=split_idx,
        )

        a = x[:, :active_c]
        b = x[:, active_c:]

        b = self.m(b)

        x = torch.cat((a, b), dim=1)

        in_idx = group_idx

        out_idx = torch.arange(
            self.cv2.active_out(),
            device=x.device,
        )

        return self.cv2.forward_indexed(
            x,
            out_idx=out_idx,
            in_idx=in_idx,
        )

    def _build_group_idx_cache(self):
        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _build_group_prefix_cache(
            module=self,
            cache_attr="_group_idx_names",
            buffer_prefix="cached_c2psa_group_idx",
            group_max=self.c,
            num_groups=2,
            widths=widths,
            device=device,
        )

    def _select_group_idx_cache(self, width):
        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _select_cached_group_prefix(
            module=self,
            cache_attr="_group_idx_names",
            buffer_prefix="cached_c2psa_group_idx",
            active_attr="_active_group_idx_name",
            group_max=self.c,
            num_groups=2,
            width=width,
            widths=widths,
            device=device,
        )

    def set_cached_group_indices(self, enabled: bool):
        self._use_cached_group_idx = bool(
            enabled
        )
        return self

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
        # At narrow widths, cache the fully materialized cv3 weight so
        # the output-prefix and grouped/non-contiguous input selections
        # are not repeated every inference.
        self._use_compact_cv3_weight_cache = True
        self._compact_cv3_weight_cache_meta = {}

    def set_width(self, width: float):

        if not hasattr(
                self,
                "_group_idx_names",
        ):
            self._build_group_idx_cache()

        self._select_group_idx_cache(
            width
        )

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

        use_group_cache = getattr(
            self,
            "_use_cached_group_idx",
            True,
        )

        if use_group_cache:

            if not hasattr(
                    self,
                    "_active_group_idx_name",
            ):
                self._build_group_idx_cache()
                self._select_group_idx_cache(
                    self.cv1.width_mult
                )

            in_idx = getattr(
                self,
                self._active_group_idx_name,
            )

        else:

            in_idx = _group_prefix_indices(
                self.c,
                active_c,
                2,
                x.device,
            )

        # ==============================================================
        # OPTIMIZATION 2B-4
        # ==============================================================

        use_compact_cache = getattr(
            self,
            "_use_compact_cv3_weight_cache",
            True,
        )

        use_materialized_weight = (
                use_compact_cache
                and not self.training
                and abs(
            self.cv3.width_mult - 1.0
        ) >= 1e-6
        )

        if use_materialized_weight:
            weight = (
                self._get_compact_cv3_weight(
                    in_idx
                )
            )

            return self.cv3.forward_materialized(
                cat,
                weight,
            )

        # --------------------------------------------------------------
        # Original path:
        #   - training
        #   - full width
        #   - controlled A/B with 2B-4 disabled
        # --------------------------------------------------------------

        out_idx = torch.arange(
            self.cv3.active_out(),
            device=x.device,
        )

        return self.cv3.forward_indexed(
            cat,
            out_idx=out_idx,
            in_idx=in_idx,
        )

    def _build_group_idx_cache(self):

        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _build_group_prefix_cache(
            module=self,
            cache_attr="_group_idx_names",
            buffer_prefix="cached_c3k_group_idx",
            group_max=self.c,
            num_groups=2,
            widths=widths,
            device=device,
        )

    def _select_group_idx_cache(self, width):

        widths = tuple(
            self.cv1.bn.widths
        )

        device = self.cv1.conv.weight.device

        _select_cached_group_prefix(
            module=self,
            cache_attr="_group_idx_names",
            buffer_prefix="cached_c3k_group_idx",
            active_attr="_active_group_idx_name",
            group_max=self.c,
            num_groups=2,
            width=width,
            widths=widths,
            device=device,
        )

    def set_cached_group_indices(self, enabled: bool):
        self._use_cached_group_idx = bool(
            enabled
        )
        return self

    def _init_compact_cv3_weight_cache(self):
        """
        Lazily initialize Optimization 2B-4 metadata.

        Required for older serialized checkpoints whose current
        __init__() is not rerun when torch.load() restores the module.
        """

        if not hasattr(
                self,
                "_compact_cv3_weight_cache_meta",
        ):
            self._compact_cv3_weight_cache_meta = {}

    def _build_compact_cv3_weight(
            self,
            in_idx,
    ):
        """
        Materialize the complete narrow-width SlimC3k.cv3 weight.

        Existing runtime path:

            full cv3 weight
                -> output prefix index_select
                -> grouped/non-contiguous input index_select
                -> convolution

        Optimization 2B-4 performs the selection once and caches the
        final compact tensor.
        """

        self._init_compact_cv3_weight_cache()

        width = float(
            self.cv3.width_mult
        )

        if abs(width - 1.0) < 1e-6:
            raise RuntimeError(
                "SlimC3k compact cv3 weight should not be built "
                "at width 1.0. Optimization 2B-1 handles the "
                "full-width path."
            )

        widths = tuple(
            self.cv3.bn.widths
        )

        key = _width_key(
            width,
            widths,
        )

        buffer_name = (
            f"_cached_c3k_cv3_weight_{key}"
        )

        device = (
            self.cv3.conv.weight.device
        )

        in_idx = torch.as_tensor(
            in_idx,
            device=device,
            dtype=torch.long,
        )

        # ----------------------------------------------------------
        # Output mapping is an active contiguous prefix.
        # ----------------------------------------------------------

        cout = self.cv3.active_out()

        weight = self.cv3.conv.weight[
            :cout,
            :,
            :,
            :,
        ]

        # ----------------------------------------------------------
        # Input corresponds to:
        #
        #   [active branch A channels,
        #    active branch B channels]
        #
        # inside the full-width two-group concatenation.
        # This is non-contiguous at narrow widths.
        # ----------------------------------------------------------

        weight = weight.index_select(
            1,
            in_idx,
        )

        # Own an inference-only compact tensor.
        weight = (
            weight
            .detach()
            .contiguous()
            .clone()
        )

        if buffer_name in self._buffers:

            self._buffers[
                buffer_name
            ] = weight

        else:

            self.register_buffer(
                buffer_name,
                weight,
                persistent=False,
            )

        self._compact_cv3_weight_cache_meta[
            key
        ] = {
            "buffer_name": buffer_name,

            "weight_version":
                self.cv3.conv.weight._version,

            "in_channels":
                int(len(in_idx)),

            "out_channels":
                int(cout),
        }

        return getattr(
            self,
            buffer_name,
        )

    def _get_compact_cv3_weight(
            self,
            in_idx,
    ):
        """
        Return a valid cached compact cv3 weight for the current width.

        Rebuild automatically if the underlying full-width parameter
        has changed.
        """

        self._init_compact_cv3_weight_cache()

        width = float(
            self.cv3.width_mult
        )

        widths = tuple(
            self.cv3.bn.widths
        )

        key = _width_key(
            width,
            widths,
        )

        meta = (
            self._compact_cv3_weight_cache_meta.get(
                key
            )
        )

        cout = self.cv3.active_out()

        cache_valid = (
                meta is not None
                and meta["buffer_name"] in self._buffers
                and meta["weight_version"]
                == self.cv3.conv.weight._version
                and meta["in_channels"]
                == int(len(in_idx))
                and meta["out_channels"]
                == int(cout)
        )

        if not cache_valid:
            return self._build_compact_cv3_weight(
                in_idx
            )

        return getattr(
            self,
            meta["buffer_name"],
        )

    def set_compact_cv3_weight_cache(
            self,
            enabled: bool,
    ):
        """
        Enable/disable Optimization 2B-4.

        enabled=True:
            Narrow-width eval inference reuses a completely
            materialized SlimC3k.cv3 convolution weight.

        enabled=False:
            Reproduce the existing forward_indexed() path.

        Width 1.0 remains handled by Optimization 2B-1.
        Training always uses the original parameter path.
        """

        self._use_compact_cv3_weight_cache = bool(
            enabled
        )

        return self

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