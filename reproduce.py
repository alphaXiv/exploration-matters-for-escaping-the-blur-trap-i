#!/usr/bin/env python3
"""Controlled multi-GPU reproduction of the ExploreGS blur-trap mechanisms."""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn


def logit(x: float) -> float:
    return math.log(x / (1.0 - x))


def psnr(mse: Tensor) -> Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


@dataclass(frozen=True)
class Camera:
    center: tuple[float, float, float]
    focal: float = 1.45


class GaussianScene(nn.Module):
    def __init__(
        self,
        xyz: Tensor,
        scales: Tensor,
        colors: Tensor,
        opacities: Tensor,
    ) -> None:
        super().__init__()
        self.xyz = nn.Parameter(xyz.clone())
        self.log_scales = nn.Parameter(scales.clone().log())
        self.color_logits = nn.Parameter(torch.logit(colors.clamp(0.01, 0.99)))
        self.opacity_logits = nn.Parameter(torch.logit(opacities.clamp(0.01, 0.99)))

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    def values(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.xyz,
            self.log_scales.exp().clamp(0.025, 1.5),
            self.color_logits.sigmoid(),
            self.opacity_logits.sigmoid(),
        )

    def render(self, cameras: list[Camera], image_size: int) -> Tensor:
        xyz, scales, colors, opacities = self.values()
        axis = torch.linspace(-1.0, 1.0, image_size, device=xyz.device, dtype=xyz.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        images: list[Tensor] = []
        for camera in cameras:
            center = xyz.new_tensor(camera.center)
            local = xyz - center
            z = local[:, 2].clamp_min(0.2)
            uv = camera.focal * local[:, :2] / z[:, None]
            radius = (camera.focal * scales / z).clamp(0.018, 0.6)
            d2 = (xx[None] - uv[:, 0, None, None]).square()
            d2 = d2 + (yy[None] - uv[:, 1, None, None]).square()
            alpha = opacities[:, None, None] * torch.exp(
                -0.5 * d2 / radius[:, None, None].square()
            )
            alpha = alpha.clamp(0.0, 0.985)
            order = torch.argsort(z)
            alpha = alpha[order]
            ordered_colors = colors[order]
            trans = torch.cumprod(
                torch.cat(
                    [torch.ones_like(alpha[:1]), (1.0 - alpha[:-1]).clamp_min(1e-5)],
                    dim=0,
                ),
                dim=0,
            )
            image = (trans[..., None] * alpha[..., None] * ordered_colors[:, None, None, :]).sum(0)
            images.append(image)
        return torch.stack(images)

    @torch.no_grad()
    def append(self, xyz: Tensor, scales: Tensor, colors: Tensor, opacities: Tensor) -> None:
        old_xyz, old_scales, old_colors, old_opacities = self.values()
        self.xyz = nn.Parameter(torch.cat([old_xyz, xyz], dim=0))
        self.log_scales = nn.Parameter(torch.cat([old_scales, scales], dim=0).log())
        self.color_logits = nn.Parameter(torch.logit(torch.cat([old_colors, colors], dim=0).clamp(0.01, 0.99)))
        self.opacity_logits = nn.Parameter(torch.logit(torch.cat([old_opacities, opacities], dim=0).clamp(0.01, 0.99)))

    @torch.no_grad()
    def split(self, indices: Tensor, generator: torch.Generator) -> None:
        if indices.numel() == 0:
            return
        xyz, scales, colors, opacities = self.values()
        chosen = torch.zeros(self.count, dtype=torch.bool, device=xyz.device)
        chosen[indices] = True
        keep = ~chosen
        parent_xyz = xyz[indices]
        parent_scales = scales[indices]
        direction = torch.randn(parent_xyz.shape, generator=generator, device=xyz.device, dtype=xyz.dtype)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        offset = direction * parent_scales[:, None] * 0.35
        child_xyz = torch.cat([parent_xyz - offset, parent_xyz + offset], dim=0)
        child_scales = torch.cat([parent_scales, parent_scales], dim=0) / 1.6
        child_colors = torch.cat([colors[indices], colors[indices]], dim=0)
        child_opacities = torch.cat([opacities[indices], opacities[indices]], dim=0) * 0.72
        self.xyz = nn.Parameter(torch.cat([xyz[keep], child_xyz], dim=0))
        self.log_scales = nn.Parameter(torch.cat([scales[keep], child_scales], dim=0).log())
        self.color_logits = nn.Parameter(torch.logit(torch.cat([colors[keep], child_colors], dim=0).clamp(0.01, 0.99)))
        self.opacity_logits = nn.Parameter(torch.logit(torch.cat([opacities[keep], child_opacities], dim=0).clamp(0.01, 0.99)))


def make_optimizer(scene: GaussianScene) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": [scene.xyz], "lr": 0.012},
            {"params": [scene.log_scales], "lr": 0.006},
            {"params": [scene.color_logits], "lr": 0.025},
            {"params": [scene.opacity_logits], "lr": 0.018},
        ]
    )


def projection_orthogonality(device: torch.device, samples: int, generator: torch.Generator) -> dict[str, float]:
    dtype = torch.float64
    xyz = torch.empty((samples, 3), device=device, dtype=dtype)
    xyz[:, :2].uniform_(-2.0, 2.0, generator=generator)
    xyz[:, 2].uniform_(2.0, 9.0, generator=generator)
    xyz.requires_grad_(True)
    target = torch.empty((samples, 2), device=device, dtype=dtype).uniform_(-0.8, 0.8, generator=generator)
    focal = 1.7
    projected = focal * xyz[:, :2] / xyz[:, 2:3]
    loss = (projected - target).square().sum()
    grad = torch.autograd.grad(loss, xyz)[0]
    rays = xyz.detach() / xyz.detach().norm(dim=1, keepdim=True)
    absolute_cos = (grad * rays).sum(1).abs() / grad.norm(dim=1).clamp_min(1e-30)
    return {
        "orthogonality_abs_cos_mean": float(absolute_cos.mean()),
        "orthogonality_abs_cos_max": float(absolute_cos.max()),
        "orthogonality_pass_rate": float((absolute_cos < 1e-10).double().mean()),
    }


def blending_attenuation(device: torch.device, batches: int, generator: torch.Generator) -> dict[str, float]:
    layers, image_size = 48, 22
    axis = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.empty((batches, layers, 2), device=device).normal_(0.0, 0.18, generator=generator)
    xy.requires_grad_(True)
    radii = torch.empty((batches, layers, 1, 1), device=device).uniform_(0.22, 0.32, generator=generator)
    opacity = torch.empty((batches, layers, 1, 1), device=device).uniform_(0.15, 0.24, generator=generator)
    colors = torch.empty((batches, layers, 1, 1, 3), device=device).uniform_(0.1, 0.95, generator=generator)
    d2 = (xx[None, None] - xy[:, :, 0, None, None]).square()
    d2 = d2 + (yy[None, None] - xy[:, :, 1, None, None]).square()
    alpha = (opacity * torch.exp(-0.5 * d2 / radii.square())).clamp_max(0.98)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), (1.0 - alpha[:, :-1]).clamp_min(1e-5)], dim=1),
        dim=1,
    )
    rendered = (trans[..., None] * alpha[..., None] * colors).sum(1)
    target = torch.zeros_like(rendered)
    loss = (rendered - target).square().mean()
    gradient = torch.autograd.grad(loss, xy)[0].norm(dim=2)
    profile = gradient.mean(0)
    early = profile[:8].mean()
    late = profile[-8:].mean()
    index = torch.arange(layers, device=device, dtype=profile.dtype)
    corr = torch.corrcoef(torch.stack([index, profile.log().clamp_min(-40)]))[0, 1]
    return {
        "attenuation_late_over_early": float(late / early.clamp_min(1e-20)),
        "attenuation_loggrad_rank_corr": float(corr),
        "attenuation_early_grad": float(early),
        "attenuation_late_grad": float(late),
    }


def far_ground_truth(device: torch.device) -> GaussianScene:
    xyz = torch.tensor(
        [
            [-1.05, -0.45, 5.8], [-0.55, 0.38, 4.8], [-0.05, -0.25, 5.4],
            [0.45, 0.40, 4.5], [0.92, -0.38, 5.6], [0.10, 0.62, 6.2],
        ], device=device,
    )
    scales = torch.tensor([0.42, 0.34, 0.30, 0.32, 0.38, 0.28], device=device)
    colors = torch.tensor(
        [[0.95, 0.16, 0.12], [0.10, 0.85, 0.25], [0.15, 0.30, 0.95],
         [0.95, 0.78, 0.10], [0.75, 0.15, 0.88], [0.10, 0.85, 0.88]], device=device,
    )
    return GaussianScene(xyz, scales, colors, torch.full((6,), 0.92, device=device))


def far_initial(device: torch.device) -> GaussianScene:
    truth = far_ground_truth(device)
    xyz, scales, colors, _ = truth.values()
    wrong_z = torch.full_like(xyz[:, 2], 2.2)
    factor = wrong_z / xyz[:, 2]
    wrong_xyz = torch.cat([xyz[:, :2] * factor[:, None], wrong_z[:, None]], dim=1)
    wrong_scales = scales * factor
    anchor_xyz = torch.tensor([[-0.5, -0.5, 1.0], [0.5, 0.5, 7.0]], device=device)
    wrong_xyz = torch.cat([wrong_xyz, anchor_xyz], dim=0)
    wrong_scales = torch.cat([wrong_scales, torch.tensor([0.08, 0.08], device=device)])
    colors = torch.cat([colors, torch.full((2, 3), 0.5, device=device)], dim=0)
    opacity = torch.cat([torch.full((6,), 0.76, device=device), torch.full((2,), 0.02, device=device)])
    return GaussianScene(wrong_xyz, wrong_scales, colors, opacity)


def near_ground_truth(device: torch.device) -> GaussianScene:
    foreground = torch.tensor([[0.0, 0.0, 2.0], [-0.38, 0.16, 2.15]], device=device)
    background = torch.tensor(
        [[-0.42, -0.34, 4.0], [0.38, -0.32, 4.0], [-0.36, 0.34, 4.0], [0.40, 0.35, 4.0]],
        device=device,
    )
    xyz = torch.cat([foreground, background], dim=0)
    scales = torch.tensor([0.68, 0.42, 0.28, 0.26, 0.27, 0.25], device=device)
    colors = torch.tensor(
        [[0.55, 0.54, 0.52], [0.22, 0.20, 0.18], [0.95, 0.18, 0.12],
         [0.10, 0.85, 0.22], [0.12, 0.28, 0.95], [0.95, 0.78, 0.10]], device=device,
    )
    opacity = torch.tensor([0.97, 0.92, 0.90, 0.90, 0.90, 0.90], device=device)
    return GaussianScene(xyz, scales, colors, opacity)


def near_initial(device: torch.device) -> GaussianScene:
    xyz = torch.tensor([[0.0, 0.0, 2.0], [-0.38, 0.16, 2.15], [0.0, 0.0, 4.0]], device=device)
    scales = torch.tensor([0.68, 0.42, 0.82], device=device)
    colors = torch.tensor([[0.55, 0.54, 0.52], [0.22, 0.20, 0.18], [0.52, 0.52, 0.40]], device=device)
    opacity = torch.tensor([0.97, 0.92, 0.78], device=device)
    return GaussianScene(xyz, scales, colors, opacity)


def add_random_seeds(scene: GaussianScene, count: int, generator: torch.Generator) -> None:
    with torch.no_grad():
        xyz, scales, colors, _ = scene.values()
        lo, hi = xyz.min(0).values, xyz.max(0).values
        span = (hi - lo).clamp_min(torch.tensor([0.4, 0.4, 2.0], device=xyz.device))
        lo = lo - 0.08 * span
        hi = hi + 0.08 * span
        new_xyz = lo + torch.rand((count, 3), generator=generator, device=xyz.device) * (hi - lo)
        source = torch.randint(0, scene.count, (count,), generator=generator, device=xyz.device)
        new_colors = colors[source]
        new_scales = scales.median().expand(count).clone()
        new_opacities = torch.full((count,), 0.08, device=xyz.device)
        scene.append(new_xyz, new_scales, new_colors, new_opacities)


def train_task(
    task: str,
    condition: str,
    steps: int,
    image_size: int,
    seed_count: int,
    split_count: int,
    split_event_fractions: list[float],
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    if task == "far":
        truth = far_ground_truth(device)
        scene = far_initial(device)
        train_cameras = [Camera((-0.14, 0.0, 0.0)), Camera((0.0, 0.0, 0.0)), Camera((0.14, 0.0, 0.0))]
        test_cameras = [Camera((-0.38, 0.03, 0.0)), Camera((0.38, -0.03, 0.0))]
    else:
        truth = near_ground_truth(device)
        scene = near_initial(device)
        train_cameras = [Camera((-0.22, 0.0, 0.0)), Camera((0.0, 0.0, 0.0)), Camera((0.22, 0.0, 0.0))]
        test_cameras = [Camera((-0.42, 0.04, 0.0)), Camera((0.42, -0.04, 0.0))]
    with torch.no_grad():
        train_target = truth.render(train_cameras, image_size)
        test_target = truth.render(test_cameras, image_size)
    optimizer = make_optimizer(scene)
    grad_ema = torch.zeros(scene.count, device=device)
    seed_enabled = task == "far" and condition in {"seed", "both"}
    split_enabled = task == "near" and condition in {"split", "both"}
    seed_event_steps = {steps // 4, steps // 2, 3 * steps // 4}
    split_event_steps = {
        max(1, min(steps, int(round(steps * fraction))))
        for fraction in split_event_fractions
    }
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = scene.render(train_cameras, image_size)
        photometric = (prediction - train_target).square().mean()
        opacity_regularizer = 2e-5 * scene.opacity_logits.sigmoid().mean()
        loss = photometric + opacity_regularizer
        loss.backward()
        with torch.no_grad():
            current_grad = scene.xyz.grad[:, :2].norm(dim=1)
            if current_grad.shape == grad_ema.shape:
                grad_ema.mul_(0.92).add_(current_grad, alpha=0.08)
        optimizer.step()
        if seed_enabled and (step + 1) in seed_event_steps:
            add_random_seeds(scene, seed_count, generator)
            optimizer = make_optimizer(scene)
            grad_ema = torch.zeros(scene.count, device=device)
        if task == "near" and (step + 1) in split_event_steps:
            with torch.no_grad():
                scales = scene.log_scales.exp()
                if split_enabled:
                    candidates = torch.topk(scales, k=min(max(split_count * 2, 1), scene.count)).indices
                    perm = torch.randperm(candidates.numel(), generator=generator, device=device)
                    chosen = candidates[perm[: min(split_count, candidates.numel())]]
                else:
                    threshold = 7.5e-5
                    eligible = torch.where((scales > 0.36) & (grad_ema > threshold))[0]
                    chosen = eligible[:split_count]
                scene.split(chosen, generator)
                optimizer = make_optimizer(scene)
                grad_ema = torch.zeros(scene.count, device=device)
    with torch.no_grad():
        train_mse = (scene.render(train_cameras, image_size) - train_target).square().mean()
        test_mse = (scene.render(test_cameras, image_size) - test_target).square().mean()
        opacities = scene.opacity_logits.sigmoid()
        effective = int((opacities > 0.05).sum())
        depth = scene.xyz[opacities > 0.05, 2]
        depth_mean = float(depth.mean()) if depth.numel() else float("nan")
    return {
        f"{task}_train_psnr": float(psnr(train_mse)),
        f"{task}_test_psnr": float(psnr(test_mse)),
        f"{task}_gaussians": float(scene.count),
        f"{task}_effective_gaussians": float(effective),
        f"{task}_mean_depth": depth_mean,
    }


def summarize(results: list[dict[str, Any]], config: dict[str, Any], elapsed: float) -> dict[str, Any]:
    numeric_keys = sorted(
        key for key, value in results[0].items() if isinstance(value, (int, float)) and key not in {"rank", "seed"}
    )
    summary: dict[str, Any] = {
        "condition": config["condition"],
        "world_size": len(results),
        "elapsed_seconds_max_rank": elapsed,
        "paper": "arXiv:2607.17965",
        "official_code_status": "unreleased_at_2026-07-21",
    }
    for key in numeric_keys:
        values = [float(item[key]) for item in results]
        summary[f"{key}_mean"] = statistics.fmean(values)
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def main() -> None:
    config = json.loads(Path("config.json").read_text())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("gloo")
    seed = int(config["base_seed"]) + 1009 * rank
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    start = time.time()
    result: dict[str, Any] = {"rank": rank, "seed": seed}
    result.update(projection_orthogonality(device, int(config["orthogonality_samples"]), generator))
    result.update(blending_attenuation(device, int(config["attenuation_batches"]), generator))
    result.update(
        train_task(
            "far", config["condition"], int(config["far_steps"]), int(config["image_size"]),
            int(config["seed_count"]), int(config["split_count"]),
            list(config.get("split_event_fractions", [0.25, 0.5, 0.75])), device, generator,
        )
    )
    result.update(
        train_task(
            "near", config["condition"], int(config["near_steps"]), int(config["image_size"]),
            int(config["seed_count"]), int(config["split_count"]),
            list(config.get("split_event_fractions", [0.25, 0.5, 0.75])), device, generator,
        )
    )
    result["elapsed_seconds"] = time.time() - start
    print("RANK_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(result, gathered, dst=0)
    else:
        gathered = [result]
    if rank == 0:
        complete = [item for item in gathered or [] if item is not None]
        elapsed = max(float(item["elapsed_seconds"]) for item in complete)
        summary = summarize(complete, config, elapsed)
        print("\n=== ORX_FINAL_SUMMARY ===")
        print("CONFIG " + json.dumps(config, sort_keys=True))
        print(f"TRIALS {len(complete)}")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"METRIC {key}={value:.10g}")
        print("METRICS_JSON " + json.dumps(summary, sort_keys=True))
        print("=== END_ORX_FINAL_SUMMARY ===", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
