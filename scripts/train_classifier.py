"""Resumable fine-grained classifier training (MobileNetV3-Large / EfficientNetV2-S / optional ConvNeXt-Tiny).

Train on the train subset, select on the validation subset (both from the official TRAIN split). The official
test split is not available to this script by design (the crop index it reads contains no test rows).

Outputs in --out (point this at persistent storage, e.g. Google Drive):
  config.json, env.json, history.csv, last.pt (resumable), best.pt (best val top-1 weights),
  val_logits_best.npz (validation logits + labels from best.pt), checkpoint_manifest.json (sha256, verified)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_recognition.classification.data import CropDataset, CropPolicy  # noqa: E402
from vehicle_recognition.classification.models import (  # noqa: E402
    build_model,
    count_params,
    param_groups,
    set_backbone_trainable,
)

NUM_CLASSES = 196


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-index", required=True, help="crops_index.csv (train/val rows only)")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--freeze-epochs", type=int, default=2, help="head-only warm start epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR; backbone uses lr * backbone-lr-mult")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--sampler", choices=["none", "weighted"], default="none")
    ap.add_argument("--class-weighted-loss", action="store_true")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--aug", default="moderate", choices=["light", "moderate", "none"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train-batches", type=int, default=0, help="smoke-test only")
    ap.add_argument("--roi-pad-frac", type=float, default=0.05)
    return ap.parse_args()


AUGS = {
    "none": {"hflip": False},
    "light": {"hflip": True, "color_jitter": (0.2, 0.2, 0.1), "box_jitter": 0.05},
    "moderate": {"hflip": True, "rotation": 8, "perspective": 0.1, "color_jitter": (0.3, 0.3, 0.2), "blur_p": 0.1,
                 "erase_p": 0.25, "box_jitter": 0.08},
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def atomic_save(obj, path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    # verify after (possibly network-backed) write
    chk = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in chk, f"checkpoint verification failed for {path}"
    return sha256(path)


def lr_lambda_factory(total_steps: int, warmup_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    return f


@torch.no_grad()
def evaluate(model, loader, device, amp) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
        logits.append(out.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    seed_all(a.seed)
    idx = pd.read_csv(a.crop_index)
    if "test" in set(idx["subset"]):
        raise SystemExit("Refusing to train: crop index contains official test rows")
    tr, va = idx[idx.subset == "train"], idx[idx.subset == "val"]
    assert tr.class_id.nunique() == NUM_CLASSES and va.class_id.nunique() == NUM_CLASSES, "class coverage check failed"
    policy = CropPolicy(roi_pad_frac=a.roi_pad_frac)
    ds_tr = CropDataset(tr, Path(a.cache_root), a.img_size, policy, train=True, aug=AUGS[a.aug], seed=a.seed)
    ds_va = CropDataset(va, Path(a.cache_root), a.img_size, policy, train=False)
    sampler = None
    if a.sampler == "weighted":
        cnt = tr.class_id.value_counts()
        w = tr.class_id.map(lambda c: 1.0 / cnt[c]).values
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                       pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=worker_init, generator=g,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size * 2, shuffle=False, num_workers=a.workers,
                       pin_memory=device.type == "cuda")

    model = build_model(a.arch, NUM_CLASSES, pretrained=True, dropout=a.dropout).to(device)
    cw = None
    if a.class_weighted_loss:
        cnt = tr.class_id.value_counts().sort_index().values.astype(np.float64)
        cw = torch.tensor(cnt.mean() / cnt, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=a.label_smoothing)
    opt = torch.optim.AdamW(param_groups(model, a.lr * a.backbone_lr_mult, a.lr, a.weight_decay))
    steps_per_epoch = len(dl_tr) if not a.max_train_batches else min(len(dl_tr), a.max_train_batches)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(a.epochs * steps_per_epoch, int(a.warmup_epochs * steps_per_epoch)))
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")

    cfg = {**vars(a), "device": str(device), "aug_params": AUGS[a.aug], "n_train": len(tr), "n_val": len(va),
           "params": count_params(model), "crop_policy": policy.__dict__}
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    start_epoch, best, bad, history = 0, -1.0, 0, []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["config"]["arch"] != a.arch or ck["config"]["img_size"] != a.img_size:
            raise SystemExit("last.pt belongs to a different configuration; use a new --out directory")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad, history = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"]
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        print(f"[resume] from epoch {start_epoch} (best val top1 {best:.4f})")
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env), "\nparams", cfg["params"], "train", len(tr), "val", len(va), "device", device)

    manifest = json.loads((out / "checkpoint_manifest.json").read_text()) if (out / "checkpoint_manifest.json").exists() else {}
    for epoch in range(start_epoch, a.epochs):
        # data order is a pure function of (seed, epoch): identical whether or not the run was resumed
        g.manual_seed(a.seed * 1000 + epoch)
        frozen = epoch < a.freeze_epochs
        set_backbone_trainable(model, not frozen)
        model.train()
        t0, tot, n, correct = time.time(), 0.0, 0, 0
        for bi, (x, y) in enumerate(dl_tr):
            if a.max_train_batches and bi >= a.max_train_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=a.amp and device.type == "cuda"):
                outp = model(x)
                loss = crit(outp, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch} batch {bi}")
            tot += loss.item() * len(y)
            n += len(y)
            correct += (outp.argmax(1) == y).sum().item()
        logits, yv = evaluate(model, dl_va, device, a.amp)
        top1 = float((logits.argmax(1) == yv).mean())
        top5 = float((np.argsort(-logits, 1)[:, :5] == yv[:, None]).any(1).mean())
        vloss = float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(yv)).item())
        rec = {"epoch": epoch, "frozen_backbone": frozen, "train_loss": tot / max(1, n), "train_acc": correct / max(1, n),
               "val_loss": vloss, "val_top1": top1, "val_top5": top5, "lr_head": opt.param_groups[2]["lr"],
               "epoch_s": round(time.time() - t0, 1)}
        history.append(rec)
        improved = top1 > best
        if improved:
            best, bad = top1, 0
            manifest["best.pt"] = {"sha256": atomic_save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                                                          "val_top1": top1}, out / "best.pt"), "epoch": epoch, "val_top1": top1}
            np.savez_compressed(out / "val_logits_best.npz", logits=logits, labels=yv,
                                relative_path=np.array(va.relative_path.tolist(), dtype=str))
        else:
            bad += 1
        manifest["last.pt"] = {"sha256": atomic_save({
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad": bad, "history": history, "config": cfg,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        }, last), "epoch": epoch}
        (out / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"ep {epoch:3d} {'F' if frozen else ' '} loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | "
              f"val loss {vloss:.3f} top1 {top1:.4f} top5 {top5:.4f} {'*' if improved else ''} ({rec['epoch_s']}s)", flush=True)
        if bad >= a.patience and not frozen:
            print(f"[early stop] no val improvement for {a.patience} epochs")
            break
    (out / "DONE").write_text(json.dumps({"best_val_top1": best, "epochs_run": len(history)}))
    print("best val top1", best)


if __name__ == "__main__":
    main()
