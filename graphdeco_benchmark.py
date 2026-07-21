"""Pinned official Graphdeco 3DGS benchmark orchestration for each torchrun rank."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch.distributed as dist


SCENES = ("train", "truck", "drjohnson", "playroom")


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("SETUP_COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _prepare_graphdeco(config: dict[str, Any], root: Path) -> None:
    source = root / "gaussian-splatting"
    data = root / "data"
    commit = str(config["graphdeco_commit"])
    if not source.exists():
        root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git", "clone", "--recursive", "https://github.com/graphdeco-inria/gaussian-splatting.git",
                str(source),
            ]
        )
    _run(["git", "checkout", commit], cwd=source)
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)

    general_utils = source / "utils" / "general_utils.py"
    text = general_utils.read_text()
    old = "import random\n"
    if "import os\n" not in text:
        text = text.replace(old, old + "import os\n", 1)
    old_seed = "    random.seed(0)\n    np.random.seed(0)\n    torch.manual_seed(0)"
    new_seed = (
        "    seed = int(os.environ.get('GS_SEED', '0'))\n"
        "    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)"
    )
    if old_seed in text:
        text = text.replace(old_seed, new_seed, 1)
        general_utils.write_text(text)

    # CUDA 12.8's stricter headers no longer make the fixed-width integer
    # types transitively visible to this upstream header.
    rasterizer_header = (
        source / "submodules" / "diff-gaussian-rasterization" /
        "cuda_rasterizer" / "rasterizer_impl.h"
    )
    rasterizer_text = rasterizer_header.read_text()
    if "#include <cstdint>" not in rasterizer_text:
        rasterizer_header.write_text("#include <cstdint>\n" + rasterizer_text)

    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "12.0"
    _run(
        [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
            "./submodules/diff-gaussian-rasterization", "./submodules/fused-ssim",
            "./submodules/simple-knn", "numpy==1.26.4", "plyfile==1.0.3",
            "opencv-python-headless==4.10.0.84",
        ],
        cwd=source,
        env=build_env,
    )

    archive = root / "tandt_db.zip"
    marker = data / ".extracted"
    if not marker.exists():
        if not archive.exists():
            print(f"DATASET_DOWNLOAD {config['graphdeco_dataset_url']}", flush=True)
            urllib.request.urlretrieve(str(config["graphdeco_dataset_url"]), archive)
        data.mkdir(parents=True, exist_ok=True)
        print(f"DATASET_EXTRACT {archive}", flush=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(data)
        marker.touch()

    # Each pod runs several trials for the same scene against one shared
    # extracted dataset. Graphdeco lazily creates points3D.ply on first use;
    # concurrent readers can otherwise observe a partially written PLY.
    ply_setup = r'''
import sys
from pathlib import Path
from scene.colmap_loader import read_points3D_binary
from scene.dataset_readers import storePly

for sparse in Path(sys.argv[1]).rglob("sparse"):
    model = sparse / "0" if (sparse / "0").is_dir() else sparse
    binary = model / "points3D.bin"
    ply = model / "points3D.ply"
    if binary.is_file():
        xyz, rgb, _ = read_points3D_binary(str(binary))
        storePly(str(ply), xyz, rgb)
        print(f"PLY_READY {ply} points={len(xyz)}", flush=True)
'''
    _run([sys.executable, "-c", ply_setup, str(data)], cwd=source)
    discovered = sorted(str(path.relative_to(data)) for path in data.rglob("sparse") if path.is_dir())
    print("GRAPHDECO_SETUP_READY commit=" + commit + " sparse_dirs=" + repr(discovered), flush=True)


def _find_scene(data_root: Path, name: str) -> Path:
    candidates = [path for path in data_root.rglob(name) if path.is_dir()]
    valid = [path for path in candidates if (path / "sparse").exists() and (path / "images").exists()]
    if not valid:
        raise FileNotFoundError(f"scene {name!r} not found under {data_root}; candidates={candidates}")
    return sorted(valid, key=lambda path: len(path.parts))[0]


def _ply_vertices(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(8192).decode("latin1", errors="ignore")
    match = re.search(r"element vertex\s+(\d+)", header)
    if not match:
        raise RuntimeError(f"could not find vertex count in {path}")
    return int(match.group(1))


def run_graphdeco_trial(
    config: dict[str, Any], rank: int, local_rank: int, world_size: int, seed: int
) -> dict[str, Any]:
    root = Path("/tmp/orx-graphdeco-benchmark")
    if local_rank == 0:
        _prepare_graphdeco(config, root)
    if world_size > 1:
        dist.barrier()

    source = root / "gaussian-splatting"
    scene_name = SCENES[rank % len(SCENES)]
    scene_path = _find_scene(root / "data", scene_name)
    output = root / f"output-rank-{rank}"
    if output.exists():
        shutil.rmtree(output)
    iterations = int(config["graphdeco_iterations"])
    resolution = int(config["graphdeco_resolution"])
    command = [
        sys.executable, "train.py", "-s", str(scene_path), "-m", str(output), "--eval",
        "-r", str(resolution), "--iterations", str(iterations), "--test_iterations", str(iterations),
        "--save_iterations", str(iterations), "--disable_viewer", "--data_device", "cpu",
        "--port", str(7000 + rank),
    ]
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    child_env["GS_SEED"] = str(seed)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONFAULTHANDLER"] = "1"
    child_env["OPENBLAS_NUM_THREADS"] = "1"
    child_env["MKL_NUM_THREADS"] = "1"
    child_env["NUMEXPR_NUM_THREADS"] = "1"
    print(
        f"GRAPHDECO_START rank={rank} local_rank={local_rank} scene={scene_name} "
        f"seed={seed} iterations={iterations} resolution={resolution}",
        flush=True,
    )
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=source,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=int(config.get("graphdeco_rank_timeout", 10800)),
    )
    elapsed = time.time() - started
    tail = "\n".join(completed.stdout.splitlines()[-240:])
    print(f"GRAPHDECO_TAIL rank={rank}\n{tail}", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Graphdeco rank {rank} failed with {completed.returncode}:\n{tail}")
    matches = re.findall(r"Evaluating test:.*?PSNR\s+(?:tensor\()?([0-9]+(?:\.[0-9]+)?)", completed.stdout)
    if not matches:
        raise RuntimeError(f"Graphdeco rank {rank} produced no test PSNR:\n{tail}")
    ply = output / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    count = _ply_vertices(ply)
    return {
        "graphdeco_scene": scene_name,
        "graphdeco_test_psnr": float(matches[-1]),
        "graphdeco_gaussians": float(count),
        "graphdeco_train_seconds": elapsed,
        "graphdeco_success": 1.0,
    }
