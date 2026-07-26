"""
Adaptive device / data-parallelism resolution.

Deliberately simple: no scheduler process, no mid-run rebalancing. The
caller passes CLI-style flags at launch, this module inspects what's
actually free right now, and returns a plan the training loop follows for
the whole run.

CLI usage pattern (wired up in pipeline/run_stage.py):
    --gpus auto|cpu|0|0,1
    --parallel-strategy auto|single|ddp
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Rough safety margin: don't claim a GPU as "free enough" unless it has at
# least this many MiB free. This is intentionally conservative and coarse --
# a real per-stage memory estimate can override it via `min_free_mib`.
_DEFAULT_MIN_FREE_MIB = 6000


@dataclass
class DevicePlan:
    device_type: str            # "cuda" or "cpu"
    device_ids: list[int] = field(default_factory=list)  # empty if cpu
    use_ddp: bool = False
    reason: str = ""

    def __str__(self) -> str:
        if self.device_type == "cpu":
            return f"CPU ({self.reason})"
        mode = "DDP" if self.use_ddp else "single-GPU"
        return f"{mode} on cuda:{self.device_ids} ({self.reason})"


def _free_mib(device_index: int) -> float:
    import torch
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
    return free_bytes / (1024 ** 2)


def resolve_device(
    gpus: str = "auto",
    parallel_strategy: str = "auto",
    min_free_mib: int = _DEFAULT_MIN_FREE_MIB,
) -> DevicePlan:
    """
    Decide what to run on, without ever assuming a GPU is safe to use just
    because it exists -- only use ones with enough currently-free VRAM, and
    only use more than one if that would not cause them to compete for
    memory on a single card.
    """
    try:
        import torch
    except ImportError:
        return DevicePlan(device_type="cpu", reason="torch not installed")

    if gpus == "cpu" or not torch.cuda.is_available():
        reason = "requested cpu" if gpus == "cpu" else "no CUDA device visible"
        return DevicePlan(device_type="cpu", reason=reason)

    n_visible = torch.cuda.device_count()
    if gpus == "auto":
        candidate_ids = list(range(n_visible))
    else:
        candidate_ids = [int(x) for x in gpus.split(",") if x.strip() != ""]

    free_by_id = {}
    for idx in candidate_ids:
        try:
            free_by_id[idx] = _free_mib(idx)
        except RuntimeError as e:
            logger.warning("Could not query GPU %d, skipping it: %s", idx, e)

    usable_ids = [i for i, free in free_by_id.items() if free >= min_free_mib]

    if not usable_ids:
        return DevicePlan(
            device_type="cpu",
            reason=f"no GPU with >= {min_free_mib} MiB free "
                    f"(saw: {free_by_id})",
        )

    if parallel_strategy == "single":
        return DevicePlan(
            device_type="cuda", device_ids=[usable_ids[0]], use_ddp=False,
            reason="parallel_strategy=single",
        )

    if parallel_strategy == "ddp":
        if len(usable_ids) < 2:
            return DevicePlan(
                device_type="cuda", device_ids=[usable_ids[0]], use_ddp=False,
                reason="ddp requested but only one GPU has enough free VRAM",
            )
        return DevicePlan(
            device_type="cuda", device_ids=usable_ids, use_ddp=True,
            reason="parallel_strategy=ddp",
        )

    # auto: only parallelize if >=2 GPUs are independently free enough that
    # running on both won't make them compete for RAM on a single card.
    if len(usable_ids) >= 2:
        return DevicePlan(
            device_type="cuda", device_ids=usable_ids, use_ddp=True,
            reason=f"auto: {len(usable_ids)} GPUs each have "
                   f">= {min_free_mib} MiB free",
        )
    return DevicePlan(
        device_type="cuda", device_ids=usable_ids, use_ddp=False,
        reason="auto: only one GPU currently free enough",
    )
