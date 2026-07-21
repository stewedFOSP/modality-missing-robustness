#!/usr/bin/env python3
"""Standalone lip+face enrollment inference for modality-missing robustness demos.

Expected inputs:
  - mix.wav: 16 kHz mixture
  - lip.npz: key 'data', shape [T, D]
  - face.npz: key 'data', shape [D] or [N, D] (first vector used if 2D)

Two packaged checkpoints (same architecture):
  checkpoints/clean/model.pth.tar
  checkpoints/occlude/model.pth.tar

Example:
  python inference.py
  python inference.py --variant clean --occlude-ratio 0.2
  python inference.py --mix path/mix.wav --lip path/lip.npz --face path/face.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config.yml"
DEFAULT_SAMPLES = REPO_ROOT / "samples"
DEFAULT_VARIANTS = ("clean", "occlude")


def apply_lip_occlusion(lip, occlude_ratio, seed=None):
    from utils.occlude import random_occlude

    lip = lip.copy()
    if occlude_ratio <= 0:
        return lip
    if seed is not None:
        np.random.seed(seed)
    ratio = float(np.random.uniform(occlude_ratio - 0.1, occlude_ratio + 0.1))
    ratio = float(np.clip(ratio, 0.0, 1.0))
    ns, vs = random_occlude(len(lip), occlude_ratio=ratio, lower=15, upper=25, exist_empty=False)
    count = 0
    for i, n in enumerate(ns):
        count += vs[i]
        lip[count : count + n] = 0
        count += n
    return lip


def load_checkpoint(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in model_info["model_state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)
    return model


def load_inputs(mix_path, lip_path, face_path, occlude_ratio=0.0, seed=None):
    mix_wav, sr = sf.read(mix_path, dtype="float32")
    if mix_wav.ndim > 1:
        mix_wav = mix_wav.mean(axis=-1)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz mix, got {sr} from {mix_path}")

    lip = np.load(lip_path)["data"].astype(np.float32)
    if lip.ndim == 3 and lip.shape[0] == 1:
        lip = lip.squeeze(0)
    if lip.ndim != 2:
        raise ValueError(f"lip expects [T, D], got {lip.shape} from {lip_path}")

    face = np.load(face_path)["data"].astype(np.float32)
    if face.ndim == 2:
        face = face[0]
    if face.ndim != 1:
        raise ValueError(f"face expects [D] or [N, D], got {face.shape} from {face_path}")

    lip = apply_lip_occlusion(lip, occlude_ratio, seed=seed)

    mix_wav = torch.from_numpy(mix_wav).float()
    ilen = torch.tensor([mix_wav.shape[-1]])
    lip = torch.from_numpy(lip).float()
    face = torch.from_numpy(face).float()
    return mix_wav, ilen, lip, face


def build_model(config_path, ckpt_path, device):
    sys.path.insert(0, str(REPO_ROOT))
    from models.network import Model  # noqa: WPS433

    with open(config_path) as rfile:
        config = yaml.safe_load(rfile)

    model = Model(**config["model_kwargs"])
    model = load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one(model, mix_wav, ilen, lip, face, device):
    mix_wav = mix_wav.to(device).unsqueeze(0)
    ilen = ilen.to(device)
    lip = lip.to(device).unsqueeze(0)
    face = face.to(device).unsqueeze(0)
    est = model(mix_wav.transpose(1, -1), ilen, lip, face)[0]  # [B, C, T]
    est = est[..., : ilen[0]].transpose(-2, -1)  # [B, T, C]
    return est.squeeze().detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Lip+face enrollment inference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variant",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=list(DEFAULT_VARIANTS),
        help="Which packaged checkpoint(s) to run",
    )
    parser.add_argument("--ckpt", type=Path, default=None, help="Override checkpoint path")
    parser.add_argument("--mix", type=Path, default=DEFAULT_SAMPLES / "mix.wav")
    parser.add_argument("--lip", type=Path, default=DEFAULT_SAMPLES / "lip.npz")
    parser.add_argument("--face", type=Path, default=DEFAULT_SAMPLES / "face.npz")
    parser.add_argument(
        "--occlude-ratio",
        type=float,
        default=0.0,
        help="Center of random lip occlusion ratio (eval uses U(r-0.1, r+0.1))",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for lip occlusion")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mix_wav, ilen, lip, face = load_inputs(
        args.mix, args.lip, args.face, occlude_ratio=args.occlude_ratio, seed=args.seed
    )
    print(
        f"Inputs: mix={tuple(mix_wav.shape)} lip={tuple(lip.shape)} face={tuple(face.shape)} "
        f"occlude_ratio={args.occlude_ratio}",
        flush=True,
    )

    jobs = []
    if args.ckpt is not None:
        jobs.append(("custom", args.ckpt.resolve()))
    else:
        for name in args.variant:
            ckpt = (REPO_ROOT / "checkpoints" / name / "model.pth.tar").resolve()
            jobs.append((name, ckpt))

    for name, ckpt_path in jobs:
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        print(f"Loading {name} from {ckpt_path} on {device} ...", flush=True)
        model = build_model(config_path, ckpt_path, device)
        est = infer_one(model, mix_wav, ilen, lip, face, device)
        out_wav = args.out_dir / f"{name}.wav"
        sf.write(out_wav, est, 16000)
        print(f"[{name}] -> {out_wav}", flush=True)


if __name__ == "__main__":
    main()
