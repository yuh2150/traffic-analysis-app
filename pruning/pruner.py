import os
import math
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
import copy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Pruning")


# ============================================================
# 1.  SPARSE TRAINING  (L1 regularization on BatchNorm gamma)
# ============================================================

def add_sparse_regularization(
    loss: torch.Tensor,
    model: nn.Module,
    sr: float = 0.0001,
) -> torch.Tensor:
    """Add L1 penalty on BatchNorm weight (gamma) to the loss.

    This is the core of Network Slimming — it drives BN gamma toward zero
    during training, so that unimportant channels can be identified and
    pruned afterward.

    Args:
        loss: Current task loss (detection loss).
        model: The model being trained.
        sr: Sparsity ratio (lambda in the paper).  Typical range 1e-5 to 1e-3.

    Returns:
        loss + sr * sum(|bn.weight|)
    """
    sparse_penalty = 0.0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            sparse_penalty += module.weight.abs().sum()
    return loss + sr * sparse_penalty


# ============================================================
# 2.  CHANNEL IMPORTANCE RANKING  (collect and sort BN gamma)
# ============================================================

def collect_bn_gammas(model: nn.Module) -> torch.Tensor:
    """Collect absolute values of all BatchNorm weight (gamma) tensors.

    Returns a flat 1-D tensor of all gamma values sorted globally.
    """
    gammas = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            gammas.append(module.weight.data.abs().view(-1))
    if not gammas:
        return torch.tensor([])
    all_gammas = torch.cat(gammas)
    return all_gammas.sort().values


def make_divisible(x: int, divisor: int = 8) -> int:
    """Round up to the nearest multiple of *divisor*.

    This ensures the pruned channel count is hardware-friendly.
    """
    return int(math.ceil(x / divisor) * divisor)


def build_pruning_plan(
    model: nn.Module,
    sparsity: float,
    divisor: int = 8,
) -> Tuple[Dict[str, List[int]], float]:
    """Build a channel pruning plan from BN gamma values.

    For each BatchNorm layer in the model, determines which channels to
    keep based on a **global** threshold computed from all gamma values.

    Returns:
        plan:       Dict[bn_module_path → list_of_kept_channel_indices]
        threshold:  The global gamma threshold used.
    """
    all_gammas = collect_bn_gammas(model)
    if all_gammas.numel() == 0:
        logger.warning("No BatchNorm layers found; returning empty plan.")
        return {}, 0.0

    threshold = torch.quantile(all_gammas, sparsity).item()
    logger.info(
        f"Global threshold = {threshold:.6f}  "
        f"(sparsity={sparsity*100:.0f}%,  "
        f"{all_gammas.numel()} gamma values)"
    )

    plan: Dict[str, List[int]] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.data.abs()
            # Keep channels whose gamma > threshold
            keep_mask = gamma > threshold
            keep_idx = torch.where(keep_mask)[0].tolist()

            # Ensure at least one channel survives
            if not keep_idx:
                best = gamma.argmax().item()
                keep_idx = [best]
                logger.debug(f"  {name}: all channels pruned, keeping 1 (ch {best})")

            # Adjust count to be a multiple of `divisor`
            num_keep = make_divisible(len(keep_idx), divisor)
            num_keep = min(num_keep, gamma.shape[0])

            # Take the top-*num_keep* channels by gamma magnitude
            _, top_idx = gamma.topk(num_keep)
            keep_idx = sorted(top_idx.tolist())

            plan[name] = keep_idx

    return plan, threshold


# ============================================================
# 3.  PHYSICAL PRUNING HELPERS  (create new modules)
# ============================================================

def _prune_conv_output(
    conv: nn.Conv2d,
    keep_indices: List[int],
) -> nn.Conv2d:
    """Create a new Conv2d with fewer output filters (structured filter prune)."""
    device = conv.weight.device
    num_keep = len(keep_indices)

    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=num_keep,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if conv.groups == 1 else num_keep,
        bias=conv.bias is not None,
    ).to(device)

    new_conv.weight.data.copy_(conv.weight.data[keep_indices])
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data[keep_indices])

    return new_conv


def _prune_bn(
    bn: nn.BatchNorm2d,
    keep_indices: List[int],
) -> nn.BatchNorm2d:
    """Create a new BatchNorm2d with fewer features (pruned channels)."""
    device = bn.weight.device
    num_keep = len(keep_indices)

    new_bn = nn.BatchNorm2d(num_keep).to(device)
    new_bn.weight.data.copy_(bn.weight.data[keep_indices])
    new_bn.bias.data.copy_(bn.bias.data[keep_indices])
    new_bn.running_mean.copy_(bn.running_mean[keep_indices])
    new_bn.running_var.copy_(bn.running_var[keep_indices])

    return new_bn


def _prune_conv_input(
    conv: nn.Conv2d,
    keep_indices: List[int],
) -> nn.Conv2d:
    """Create a new Conv2d with fewer input channels.

    Used when the *previous* layer's output channels have been pruned.
    """
    device = conv.weight.device
    num_keep = len(keep_indices)

    new_conv = nn.Conv2d(
        in_channels=num_keep,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if conv.groups == 1 else 1,  # groups may change
        bias=conv.bias is not None,
    ).to(device)

    # Copy only the kept input channels
    new_conv.weight.data.copy_(conv.weight.data[:, keep_indices])
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)

    return new_conv


# ============================================================
# 4.  YOLOv5 MODULE REBUILDING
# ============================================================

def _rebuild_conv(
    module: nn.Module,
    input_keep: Optional[List[int]],
    output_keep: Optional[List[int]],
) -> nn.Module:
    """Rebuild a YOLOv5 |Conv| module with pruned input/output channels.

    The module is expected to have attributes:  .conv (Conv2d), .bn (BN), .act.
    """
    device = module.conv.weight.device

    # --- prune input channels (because previous layer was pruned) ---
    if input_keep is not None and len(input_keep) != module.conv.in_channels:
        in_conv = _prune_conv_input(module.conv, input_keep)
    else:
        in_conv = module.conv

    # --- prune output channels (based on this BN's gamma) ---
    if output_keep is not None:
        out_conv = _prune_conv_output(in_conv, output_keep)
        out_bn = _prune_bn(module.bn, output_keep) if hasattr(module, "bn") and module.bn is not None else module.bn
    else:
        out_conv = in_conv
        out_bn = module.bn

    # --- rebuild the Conv wrapper ---
    new_mod = type(module)(out_conv.in_channels, out_conv.out_channels,
                           out_conv.kernel_size[0] if isinstance(out_conv.kernel_size, tuple) else out_conv.kernel_size,
                           out_conv.stride[0] if isinstance(out_conv.stride, tuple) else out_conv.stride,
                           out_conv.padding[0] if isinstance(out_conv.padding, tuple) else out_conv.padding,
                           out_conv.groups,
                           isinstance(module.act, nn.SiLU) if hasattr(module, "act") else True).to(device)

    new_mod.conv = out_conv
    new_mod.bn = out_bn
    if hasattr(module, "act") and hasattr(new_mod, "act"):
        new_mod.act = module.act

    return new_mod


def _rebuild_bottleneck(
    bottleneck: nn.Module,
    keep: List[int],
) -> nn.Module:
    """Rebuild a YOLOv5 |Bottleneck| with pruned internal channels.

    All internal Conv modules share the same hidden channel space,
    so they all prune to the same |keep| indices.
    """
    device = next(bottleneck.parameters()).device
    shortcut = getattr(bottleneck, "add", False)
    # c1 → c_ and c_ → c2  (c1 == c2 == hidden dim after cv1 pruning)
    c_ = len(keep)
    c_in = bottleneck.cv1.conv.in_channels

    new_cv1 = type(bottleneck.cv1)(c_in, c_,
                                    bottleneck.cv1.conv.kernel_size[0],
                                    bottleneck.cv1.conv.stride[0],
                                    bottleneck.cv1.conv.padding[0],
                                    bottleneck.cv1.conv.groups,
                                    True).to(device)
    new_cv1.conv = _prune_conv_output(bottleneck.cv1.conv, keep)
    new_cv1.bn = _prune_bn(bottleneck.cv1.bn, keep)
    new_cv1.act = bottleneck.cv1.act

    new_cv2 = type(bottleneck.cv2)(c_, c_,
                                    bottleneck.cv2.conv.kernel_size[0],
                                    bottleneck.cv2.conv.stride[0],
                                    bottleneck.cv2.conv.padding[0],
                                    bottleneck.cv2.conv.groups,
                                    True).to(device)
    new_cv2.conv = _prune_conv_input(bottleneck.cv2.conv, keep)
    new_cv2.conv = _prune_conv_output(new_cv2.conv, keep)
    new_cv2.bn = _prune_bn(bottleneck.cv2.bn, keep)
    new_cv2.act = bottleneck.cv2.act

    new_bn = type(bottleneck)(c_, c_, shortcut, bottleneck.cv2.conv.groups, 1.0).to(device)
    new_bn.cv1 = new_cv1
    new_bn.cv2 = new_cv2
    new_bn.add = shortcut

    return new_bn


def _rebuild_c3(
    c3: nn.Module,
    plan: Dict[str, List[int]],
    path: str,
    input_keep: Optional[List[int]] = None,
) -> nn.Module:
    """Rebuild a YOLOv5 |C3| module with pruned internal channels.

    Within a C3 all parallel hidden paths (cv1, cv2, bottlenecks) must
    prune to the *same* hidden channel count so that the final concat
    remains valid.
    """
    device = next(c3.parameters()).device

    # --- collect BN paths inside this C3 ---
    bn_paths = [f"{path}.cv1.bn", f"{path}.cv2.bn"]
    for j in range(len(c3.m)):
        bn_paths.append(f"{path}.m.{j}.cv1.bn")
        bn_paths.append(f"{path}.m.{j}.cv2.bn")

    # --- gather all gamma values from the hidden-space BNs ---
    hidden_gammas = []
    hidden_bns = []
    for bp in bn_paths:
        if bp in plan:
            # Use the already-planned indices for this BN
            pass
        # Find the actual BN module via path traversal
        parts = bp.split(".")
        m = c3
        for p in parts[1:]:  # skip the parent path prefix
            if p.isdigit():
                m = m[int(p)]
            else:
                m = getattr(m, p)
        if isinstance(m, nn.BatchNorm2d):
            hidden_gammas.append(m.weight.data.abs())
            hidden_bns.append((bp, m))

    # --- unify hidden ---
    if hidden_gammas:
        all_hidden = torch.cat([g.view(-1) for g in hidden_gammas])
        _, top_idx = all_hidden.topk(min(make_divisible(int(all_hidden.numel() * 0.7), 8), all_hidden.numel()))
        keep_set = set()
        for _, bn_mod in hidden_bns:
            gamma = bn_mod.weight.data.abs()
            local_thresh = gamma[top_idx[top_idx < gamma.numel()]].min() if top_idx.numel() > 0 else 0
            keep = gamma > local_thresh
            if keep.sum() < 1:
                keep = gamma >= gamma.sort().values[-1]
            for idx in torch.where(keep)[0].tolist():
                keep_set.add(idx)
        keep_hidden = sorted(keep_set)[:min(len(keep_set), hidden_gammas[0].shape[0])]
        if not keep_hidden:
            keep_hidden = [0]
        num_keep = min(hidden_gammas[0].shape[0], max(1, make_divisible(len(keep_hidden), 8)))
        _, top = hidden_gammas[0].topk(num_keep)
        keep_hidden = sorted(top.tolist())
    else:
        keep_hidden = list(range(c3.cv1.conv.out_channels))

    # --- prune cv1 (input → hidden) ---
    if input_keep is not None:
        cv1_conv = _prune_conv_input(c3.cv1.conv, input_keep)
    else:
        cv1_conv = c3.cv1.conv
    cv1_conv = _prune_conv_output(cv1_conv, keep_hidden)
    cv1_bn = _prune_bn(c3.cv1.bn, keep_hidden)

    new_cv1 = type(c3.cv1)(cv1_conv.in_channels, cv1_conv.out_channels,
                            cv1_conv.kernel_size[0], cv1_conv.stride[0],
                            cv1_conv.padding[0], cv1_conv.groups, True).to(device)
    new_cv1.conv = cv1_conv
    new_cv1.bn = cv1_bn
    new_cv1.act = c3.cv1.act

    # --- prune cv2 (input → hidden, parallel shortcut path) ---
    if input_keep is not None:
        cv2_conv = _prune_conv_input(c3.cv2.conv, input_keep)
    else:
        cv2_conv = c3.cv2.conv
    cv2_conv = _prune_conv_output(cv2_conv, keep_hidden)
    cv2_bn = _prune_bn(c3.cv2.bn, keep_hidden)

    new_cv2 = type(c3.cv2)(cv2_conv.in_channels, cv2_conv.out_channels,
                            cv2_conv.kernel_size[0], cv2_conv.stride[0],
                            cv2_conv.padding[0], cv2_conv.groups, True).to(device)
    new_cv2.conv = cv2_conv
    new_cv2.bn = cv2_bn
    new_cv2.act = c3.cv2.act

    # --- rebuild bottlenecks ---
    new_bottlenecks = []
    for j, bneck in enumerate(c3.m):
        new_bneck = _rebuild_bottleneck(bneck, keep_hidden)
        new_bottlenecks.append(new_bneck)

    # --- prune cv3 (2*hidden → output) ---
    cv3 = c3.cv3
    cv3_path = f"{path}.cv3.bn"
    cv3_output_keep = plan.get(cv3_path, None)

    cv3_in = 2 * len(keep_hidden)
    cv3_conv = nn.Conv2d(cv3_in, cv3.conv.out_channels,
                          cv3.conv.kernel_size, cv3.conv.stride,
                          cv3.conv.padding, groups=cv3.conv.groups,
                          bias=cv3.conv.bias is not None).to(device)
    # We cannot directly copy weights because input channels changed.
    # Instead, expand the original weights: each output filter gets contributions
    # from cv3.in_channels in the original; now it's cv3_in.
    # We'll initialize from scratch (pruned fine-tuning will recover accuracy).
    nn.init.kaiming_normal_(cv3_conv.weight, mode="fan_out", nonlinearity="relu")
    if cv3_conv.bias is not None:
        cv3_conv.bias.data.zero_()

    if cv3_output_keep is not None:
        cv3_conv = _prune_conv_output(cv3_conv, cv3_output_keep)
        cv3_bn = _prune_bn(c3.cv3.bn, cv3_output_keep)
    else:
        cv3_bn = c3.cv3.bn

    new_cv3 = type(c3.cv3)(cv3_conv.in_channels, cv3_conv.out_channels,
                            cv3_conv.kernel_size[0], cv3_conv.stride[0],
                            cv3_conv.padding[0], cv3_conv.groups, True).to(device)
    new_cv3.conv = cv3_conv
    new_cv3.bn = cv3_bn
    new_cv3.act = c3.cv3.act

    # --- rebuild C3 ---
    c1 = c3.cv1.conv.in_channels
    c2 = cv3_conv.out_channels
    n = len(c3.m)
    e = cv3_conv.in_channels / (2 * len(keep_hidden)) if cv3_conv.in_channels > 0 else 0.5
    shortcut = getattr(c3.m[0], "add", False) if len(c3.m) > 0 else True

    new_c3 = type(c3)(c1, c2, n, shortcut, c3.m[0].cv2.conv.groups if len(c3.m) > 0 else 1, e).to(device)
    new_c3.cv1 = new_cv1
    new_c3.cv2 = new_cv2
    new_c3.cv3 = new_cv3
    new_c3.m = nn.Sequential(*new_bottlenecks)

    return new_c3


def _rebuild_sppf(
    sppf: nn.Module,
    plan: Dict[str, List[int]],
    path: str,
    input_keep: Optional[List[int]] = None,
) -> nn.Module:
    """Rebuild a YOLOv5 SPPF module with pruned channels."""
    device = next(sppf.parameters()).device

    # --- cv1 ---
    if input_keep is not None:
        cv1_conv = _prune_conv_input(sppf.cv1.conv, input_keep)
    else:
        cv1_conv = sppf.cv1.conv
    cv1_path = f"{path}.cv1.bn"
    cv1_keep = plan.get(cv1_path, None)
    if cv1_keep is not None:
        cv1_conv = _prune_conv_output(cv1_conv, cv1_keep)
        cv1_bn = _prune_bn(sppf.cv1.bn, cv1_keep)
    else:
        cv1_bn = sppf.cv1.bn

    new_cv1 = type(sppf.cv1)(cv1_conv.in_channels, cv1_conv.out_channels,
                               cv1_conv.kernel_size[0], cv1_conv.stride[0],
                               cv1_conv.padding[0], cv1_conv.groups, True).to(device)
    new_cv1.conv = cv1_conv
    new_cv1.bn = cv1_bn
    new_cv1.act = sppf.cv1.act

    # --- cv2 ---
    c_ = cv1_conv.out_channels
    cv2_conv = _prune_conv_input(sppf.cv2.conv, list(range(c_)))
    cv2_path = f"{path}.cv2.bn"
    cv2_keep = plan.get(cv2_path, None)
    if cv2_keep is not None:
        cv2_conv = _prune_conv_output(cv2_conv, cv2_keep)
        cv2_bn = _prune_bn(sppf.cv2.bn, cv2_keep)
    else:
        cv2_bn = sppf.cv2.bn

    new_cv2 = type(sppf.cv2)(cv2_conv.in_channels, cv2_conv.out_channels,
                               cv2_conv.kernel_size[0], cv2_conv.stride[0],
                               cv2_conv.padding[0], cv2_conv.groups, True).to(device)
    new_cv2.conv = cv2_conv
    new_cv2.bn = cv2_bn
    new_cv2.act = sppf.cv2.act

    # --- rebuild SPPF ---
    c1 = sppf.cv1.conv.in_channels
    c2 = cv2_conv.out_channels
    k = sppf.k if hasattr(sppf, "k") else 5

    new_sppf = type(sppf)(c1, c2, k).to(device)
    new_sppf.cv1 = new_cv1
    new_sppf.cv2 = new_cv2

    return new_sppf


# ============================================================
# 5.  NETWORK SLIMMING CHANNEL PRUNE  (main entry point)
# ============================================================

def network_slimming_channel_prune(
    model: nn.Module,
    sparsity: float = 0.5,
    divisor: int = 8,
) -> nn.Module:
    """Perform BN-gamma-based Network Slimming channel pruning on a YOLOv5 model.

    Pipeline:
        1. Collect all BN gamma values → global threshold.
        2. For each BN, decide which channels to keep (gamma > threshold).
        3. Adjust kept counts to be multiples of *divisor*.
        4. Rebuild the model — physically remove pruned channels
           by creating new Conv/BN modules with reduced dimensions.
        5. Return the pruned model with surviving weights copied.

    Args:
        model:    A YOLOv5 model (from ``torch.hub.load`` or ``YOLOWrapper``).
        sparsity: Fraction of channels to prune (0.0 – 1.0).  Default 0.5.
        divisor:  Channel counts are rounded up to multiples of this.  Default 8.

    Returns:
        A new model with pruned architecture and copied surviving weights.
    """
    model = copy.deepcopy(model)

    # Step 1 & 2: build pruning plan from BN gamma
    plan, threshold = build_pruning_plan(model, sparsity, divisor)
    if not plan:
        logger.warning("Empty pruning plan; returning unmodified model.")
        return model

    # Step 3: get the Sequential backbone
    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        model_seq = model.model
    else:
        model_seq = model

    logger.info(f"Rebuilding {len(model_seq)} modules with pruned channels ...")

    # Step 4: rebuild each module
    new_modules = []
    prev_output_keep: Optional[List[int]] = None  # input keep for next layer

    for i, module in enumerate(model_seq):
        path = f"model.{i}"
        bn_path_model = f"model.model.{i}"

        if hasattr(module, "conv") and hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
            # ----- Conv (Conv2d + BN + activation) -----
            bn_name = f"{bn_path_model}.bn"
            output_keep = plan.get(bn_name, None)

            new_mod = _rebuild_conv(module, prev_output_keep, output_keep)
            new_modules.append(new_mod)

            # Track output channels for downstream propagation
            if output_keep is not None:
                prev_output_keep = output_keep
            else:
                prev_output_keep = list(range(module.conv.out_channels))

        elif hasattr(module, "cv1") and hasattr(module, "cv2") and hasattr(module, "cv3"):
            # ----- C3 -----
            prefix = f"{bn_path_model}"
            new_mod = _rebuild_c3(module, plan, prefix, prev_output_keep)
            new_modules.append(new_mod)

            # Track output from cv3
            cv3_out = plan.get(f"{prefix}.cv3.bn", None)
            if cv3_out is not None:
                prev_output_keep = cv3_out
            else:
                prev_output_keep = list(range(module.cv3.conv.out_channels))

        elif hasattr(module, "cv1") and hasattr(module, "cv2") and not hasattr(module, "cv3"):
            # ----- SPPF -----
            prefix = f"{bn_path_model}"
            new_mod = _rebuild_sppf(module, plan, prefix, prev_output_keep)
            new_modules.append(new_mod)

            cv2_out = plan.get(f"{prefix}.cv2.bn", None)
            if cv2_out is not None:
                prev_output_keep = cv2_out
            else:
                prev_output_keep = list(range(module.cv2.conv.out_channels))

        elif isinstance(module, nn.Upsample):
            # Upsample does not change channel count
            new_modules.append(copy.deepcopy(module))

        elif module.__class__.__name__ == "Concat":
            # Concat concatenates along dim=1; output channels = sum of inputs
            # If all inputs have been pruned consistently this still works
            new_modules.append(copy.deepcopy(module))

        elif isinstance(module, nn.Identity):
            new_modules.append(module)

        elif hasattr(module, "m") and all(isinstance(m, nn.Conv2d) for m in module.m):
            # ----- Detect module (final detection head) -----
            detect_path = f"{bn_path_model}"
            new_detect = _rebuild_detect(module, plan, detect_path, prev_output_keep)
            new_modules.append(new_detect)
            # Output of Detect is not fed to any subsequent Conv

        else:
            logger.debug(f"  Layer {i} ({type(module).__name__}): passed through unchanged")
            new_modules.append(copy.deepcopy(module))

    model.model = nn.Sequential(*new_modules)

    # Update stride attribute if present
    if hasattr(model, "stride"):
        if hasattr(model_seq[0], "conv") and hasattr(model_seq[0].conv, "stride"):
            s = model_seq[0].conv.stride[0]
        else:
            s = 32
        model.stride = torch.tensor([s], dtype=torch.int)

    # Count removed channels
    total_before = sum(b.weight.numel() for _, m in model.named_modules() if isinstance(m, nn.BatchNorm2d))
    total_after = sum(len(v) for v in plan.values())
    removal_pct = (1 - total_after / max(total_before, 1)) * 100
    params_removed = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Channel pruning done.  "
        f"BN channels: {total_before} → {total_after}  "
        f"({removal_pct:.1f}% removed).  "
        f"Remaining params: {params_removed:,}"
    )

    return model


def _rebuild_detect(
    detect: nn.Module,
    plan: Dict[str, List[int]],
    path: str,
    input_keep: Optional[List[int]] = None,
) -> nn.Module:
    """Rebuild the Detect module.

    The Detect module's conv layers ('m') have no BN, but their input
    channels may have been pruned by the preceding layers.

    YOLOv5 Detect output channels = na * (5 + nc) where na = number of
    anchors per scale and nc = number of classes.  These are NOT pruned
    (class count stays the same).  Only input channels are adjusted.
    """
    device = next(detect.parameters()).device

    new_m = nn.ModuleList()
    for j, conv in enumerate(detect.m):
        if input_keep is not None:
            new_conv = _prune_conv_input(conv, input_keep)
        else:
            new_conv = copy.deepcopy(conv)
        new_m.append(new_conv)

    # Rebuild Detect — we need to use the same class
    na = detect.na
    nc = detect.nc
    channels = [c.in_channels for c in new_m]

    new_detect = type(detect)(nc, detect.anchors, channels).to(device)
    new_detect.m = new_m
    new_detect.stride = detect.stride
    new_detect.anchors = detect.anchors
    new_detect.anchor_grid = detect.anchor_grid if hasattr(detect, "anchor_grid") else detect.anchors.clone().view(-1, 1, 1, 2)

    return new_detect


# ============================================================
# 6.  LEGACY PRUNING FUNCTIONS  (unchanged)
# ============================================================

def prune_by_magnitude(model: nn.Module, sparsity: float) -> nn.Module:
    model = copy.deepcopy(model)
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_weights.append(module.weight.data.abs().view(-1))
    if not all_weights:
        return model
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, sparsity).item()
    logger.info(f"Magnitude pruning: threshold={threshold:.6f} sparsity={sparsity*100}%")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = module.weight.data.abs() >= threshold
            module.weight.data.mul_(mask.float())
            module.register_buffer("pruning_mask", mask)
    return model


def prune_by_l1_norm(model: nn.Module, sparsity: float) -> nn.Module:
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            l1_norms = w.abs().sum(dim=[1, 2, 3])
            num_filters = l1_norms.size(0)
            num_to_prune = int(num_filters * sparsity)
            if num_to_prune == 0:
                continue
            threshold = torch.topk(l1_norms, num_to_prune, largest=False).values[-1].item()
            mask = l1_norms >= threshold
            mask_expanded = mask.view(-1, 1, 1, 1).to(w.device)
            module.weight.data.mul_(mask_expanded.float())
            if module.bias is not None:
                module.bias.data.mul_(mask.float())
            module.register_buffer("filter_mask", mask)
    logger.info(f"L1-Norm filter pruning applied at {sparsity*100}% sparsity.")
    return model


def physical_filter_prune(conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d], keep_indices: List[int]) -> Tuple[nn.Conv2d, Optional[nn.BatchNorm2d]]:
    device = conv.weight.device
    num_keep = len(keep_indices)
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=num_keep,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups if conv.groups == 1 else num_keep,
        bias=conv.bias is not None,
    ).to(device)
    new_conv.weight.data.copy_(conv.weight.data[keep_indices])
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data[keep_indices])

    new_bn = None
    if bn is not None:
        new_bn = nn.BatchNorm2d(num_features=num_keep).to(device)
        new_bn.weight.data.copy_(bn.weight.data[keep_indices])
        new_bn.bias.data.copy_(bn.bias.data[keep_indices])
        new_bn.running_mean.copy_(bn.running_mean[keep_indices])
        new_bn.running_var.copy_(bn.running_var[keep_indices])
    return new_conv, new_bn


def physical_channel_prune(conv_or_linear: nn.Module, keep_indices: List[int]) -> nn.Module:
    device = next(conv_or_linear.parameters()).device
    num_keep = len(keep_indices)
    if isinstance(conv_or_linear, nn.Conv2d):
        new_layer = nn.Conv2d(
            in_channels=num_keep,
            out_channels=conv_or_linear.out_channels,
            kernel_size=conv_or_linear.kernel_size,
            stride=conv_or_linear.stride,
            padding=conv_or_linear.padding,
            dilation=conv_or_linear.dilation,
            groups=conv_or_linear.groups if conv_or_linear.groups == 1 else num_keep,
            bias=conv_or_linear.bias is not None,
        ).to(device)
        new_layer.weight.data.copy_(conv_or_linear.weight.data[:, keep_indices])
        if conv_or_linear.bias is not None:
            new_layer.bias.data.copy_(conv_or_linear.bias.data)
    elif isinstance(conv_or_linear, nn.Linear):
        new_layer = nn.Linear(
            in_features=num_keep,
            out_features=conv_or_linear.out_features,
            bias=conv_or_linear.bias is not None,
        ).to(device)
        new_layer.weight.data.copy_(conv_or_linear.weight.data[:, keep_indices])
        if conv_or_linear.bias is not None:
            new_layer.bias.data.copy_(conv_or_linear.bias.data)
    return new_layer


def prune_layer(model: nn.Module, layer_name: str) -> nn.Module:
    model = copy.deepcopy(model)
    parts = layer_name.split(".")
    curr_mod = model
    for part in parts[:-1]:
        if part.isdigit():
            curr_mod = curr_mod[int(part)]
        else:
            curr_mod = getattr(curr_mod, part)
    target_part = parts[-1]
    if target_part.isdigit():
        curr_mod[int(target_part)] = nn.Identity()
    else:
        setattr(curr_mod, target_part, nn.Identity())
    logger.info(f"Layer pruning: {layer_name} replaced with Identity.")
    return model


# --- CHANNEL DEPENDENCY TRACKING ---

def trace_dependencies(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Build a dependency graph of Conv2d layers with their connected BatchNorm layers.
    
    Returns dict mapping module paths to:
      - 'conv': nn.Conv2d module
      - 'bn': nn.BatchNorm2d or None
      - 'next_conv': list of (path, nn.Conv2d) that take this layer's output as input
    """
    module_graph: Dict[str, Dict[str, Any]] = {}

    def _channels_out(path: str, mod: nn.Module) -> int:
        if isinstance(mod, nn.Conv2d):
            return mod.out_channels
        if isinstance(mod, nn.BatchNorm2d):
            return mod.num_features
        return -1

    def _channels_in(path: str, mod: nn.Module) -> int:
        if isinstance(mod, nn.Conv2d):
            return mod.in_channels
        if isinstance(mod, nn.BatchNorm2d):
            return mod.num_features
        return -1

    conv_paths = {}
    path_to_mod = {}

    for name, module in model.named_modules():
        path_to_mod[name] = module
        if isinstance(module, nn.Conv2d):
            conv_paths[name] = {"conv": module, "bn": None}
            module_graph[name] = {"conv": module, "bn": None, "next_conv": []}

    for name, module in model.named_modules():
        path = name
        if isinstance(module, nn.BatchNorm2d):
            path_parts = path.split(".")
            if path_parts:
                parent_path = ".".join(path_parts[:-1])
                child_name = path_parts[-1]
                parent_mod = path_to_mod.get(parent_path, model)
                siblings = []
                for cname, cmod in parent_mod.named_children():
                    if isinstance(cmod, nn.Conv2d):
                        siblings.append((f"{parent_path}.{cname}" if parent_path else cname, cmod))
                if siblings:
                    for spath, smod in siblings:
                        if _channels_in("", module) == smod.out_channels:
                            if smod.out_channels == module.num_features:
                                conv_paths[spath]["bn"] = module
                                if spath in module_graph:
                                    module_graph[spath]["bn"] = module

    for path, info in module_graph.items():
        conv = info["conv"]
        out_ch = conv.out_channels
        for other_path, other_info in module_graph.items():
            if other_path == path:
                continue
            other_conv = other_info["conv"]
            if _channels_in(other_path, other_conv) == out_ch:
                info["next_conv"].append((other_path, other_conv))
                break

    return module_graph


def get_effective_metrics(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)) -> Dict[str, Any]:
    """Counts effective (non-zero) parameters and estimates FLOPs."""
    active_params = 0
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_params += p.numel()
            active_params += (p.data != 0).sum().item()

    flops_dict = {"flops": 0}

    def conv_hook(module, input, output):
        c_in = module.in_channels
        c_out = module.out_channels
        kh, kw = module.kernel_size
        batch_size = output.shape[0]
        h_out, w_out = output.shape[2], output.shape[3]
        total_w = module.weight.numel()
        active_w = (module.weight.data != 0).sum().item()
        w_sparsity = active_w / total_w if total_w > 0 else 0.0
        flops_dict["flops"] += int(2 * batch_size * c_in * c_out * kh * kw * h_out * w_out * w_sparsity)

    def linear_hook(module, input, output):
        features_in = module.in_features
        features_out = module.out_features
        batch_size = output.shape[0]
        total_w = module.weight.numel()
        active_w = (module.weight.data != 0).sum().item()
        w_sparsity = active_w / total_w if total_w > 0 else 0.0
        flops_dict["flops"] += int(2 * batch_size * features_in * features_out * w_sparsity)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    x = torch.zeros(input_size, device=next(model.parameters()).device)
    try:
        with torch.no_grad():
            model.forward(x)
    except Exception:
        flops_dict["flops"] = int(active_params * 2 * 10)
    finally:
        for h in hooks:
            h.remove()

    return {
        "total_params": total_params,
        "active_params": int(active_params),
        "sparsity": 1.0 - (active_params / total_params),
        "flops": flops_dict["flops"],
    }
