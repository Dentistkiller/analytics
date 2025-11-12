#!/usr/bin/env python3
# face_kaggle_cnn.py
# Single-file script: downloads a Kaggle face dataset (default: LFW), builds splits, trains a CNN, and can enroll/recognize.
# Requirements:
#   pip install kaggle torch torchvision pillow numpy scikit-learn opencv-python tqdm
# Kaggle API setup:
#   Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars.

import argparse, json, os, sys, time, math, pickle, glob, shutil, random, zipfile, pathlib, io
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ----------------------------
# Utilities
# ----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def is_image(p: str) -> bool:
    p = p.lower()
    return any(p.endswith(ext) for ext in [".jpg",".jpeg",".png",".bmp",".webp"])

# ----------------------------
# Model
# ----------------------------

class FaceEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embed = nn.Linear(in_feats, embed_dim)
        self.head  = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        emb   = self.embed(feats)
        emb   = nn.functional.normalize(emb, p=2, dim=1)
        logits = self.head(emb)
        return emb, logits

# ----------------------------
# Kaggle Download & Prepare
# ----------------------------

def ensure_kaggle_creds():
    cred_ok = False
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        cred_ok = True
    elif os.path.isfile(os.path.expanduser("~/.kaggle/kaggle.json")):
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
        cred_ok = True
    if not cred_ok:
        print("[ERR] Kaggle API credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or place kaggle.json in ~/.kaggle/", file=sys.stderr)
        sys.exit(2)

def kaggle_download(dataset: str, out_dir: str):
    ensure_kaggle_creds()
    from kaggle.api.kaggle_api_extended import KaggleApi
    safe_makedirs(out_dir)
    api = KaggleApi()
    api.authenticate()
    print(f"[INFO] Downloading Kaggle dataset: {dataset}")
    api.dataset_download_files(dataset, path=out_dir, quiet=False, force=False, unzip=False)

    # Find the .zip file (Kaggle API saves as <slug>.zip)
    zips = [p for p in os.listdir(out_dir) if p.endswith(".zip")]
    if not zips:
        print("[ERR] No zip file downloaded from Kaggle.", file=sys.stderr)
        sys.exit(3)
    zip_path = os.path.join(out_dir, zips[0])
    print(f"[INFO] Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    return out_dir

def detect_id_folder(root: str) -> Optional[str]:
    # Heuristic: return a folder containing many subfolders each with images (e.g., LFW)
    best = None
    best_count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if dirnames and not filenames:
            # Check if children are identity folders
            sample = dirnames[:50]
            sub_ok = 0
            for d in sample:
                dpath = os.path.join(dirpath, d)
                imgs = [f for f in os.listdir(dpath) if is_image(f)]
                if imgs:
                    sub_ok += 1
            if sub_ok >= max(5, len(sample)//2):
                # Count total identities
                total_ids = len(dirnames)
                if total_ids > best_count:
                    best = dirpath
                    best_count = total_ids
    return best

def stratified_split_id_folder(id_root: str, out_root: str, val_ratio: float = 0.2, min_images_per_id: int = 2, seed: int = 42):
    rng = random.Random(seed)
    # Collect per-identity images
    identities = []
    for person in sorted(os.listdir(id_root)):
        pdir = os.path.join(id_root, person)
        if not os.path.isdir(pdir): continue
        imgs = [os.path.join(pdir, f) for f in os.listdir(pdir) if is_image(f)]
        if len(imgs) >= min_images_per_id:
            identities.append((person, imgs))

    if len(identities) < 2:
        print("[ERR] Not enough identities with sufficient images.", file=sys.stderr)
        sys.exit(4)

    train_dir = os.path.join(out_root, "train")
    val_dir   = os.path.join(out_root, "val")
    safe_makedirs(train_dir)
    safe_makedirs(val_dir)

    # Split each identity's images into train/val
    for person, imgs in identities:
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]
        # Ensure at least 1 train image
        if len(train_imgs) == 0 and val_imgs:
            train_imgs = [val_imgs.pop()]
        # Copy
        for dst_base, subset in [(train_dir, train_imgs), (val_dir, val_imgs)]:
            dst = os.path.join(dst_base, person)
            safe_makedirs(dst)
            for src in subset:
                shutil.copy2(src, os.path.join(dst, os.path.basename(src)))
    print(f"[INFO] Split complete -> {out_root} (train/val)")

# ----------------------------
# Training / Evaluation
# ----------------------------

@dataclass
class TrainConfig:
    data_root: str
    out_dir: str
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    img_size: int = 224
    num_workers: int = 4
    seed: int = 42
    save_every: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def save_class_map(out_dir: str, class_to_idx: Dict[str,int]):
    safe_makedirs(out_dir)
    with open(os.path.join(out_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

def load_datasets(data_root: str, img_size: int) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, eval_tf = make_transforms(img_size)
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"[ERR] Expecting {data_root}/train and {data_root}/val", file=sys.stderr)
        sys.exit(2)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    return train_ds, val_ds

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, logits = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    acc = correct / total if total > 0 else 0.0
    cm = confusion_matrix(all_labels, all_preds) if total > 0 else None
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=False) if total > 0 else ""
    return loss_sum / total, acc, cm, report

def train_cmd(args):
    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.workers,
        seed=args.seed,
        save_every=args.save_every,
    )
    seed_all(cfg.seed)
    device = cfg.device
    print(f"[INFO] Using device: {device}")

    train_ds, val_ds = load_datasets(cfg.data_root, cfg.img_size)
    save_class_map(cfg.out_dir, train_ds.class_to_idx)
    num_classes = len(train_ds.classes)
    print(f"[INFO] Classes: {num_classes} -> {train_ds.classes}")

    model = FaceEmbedder(num_classes=num_classes, embed_dim=128, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    best_acc = 0.0
    safe_makedirs(cfg.out_dir)
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, cm, report = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(f"[E{epoch:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f} ({elapsed:.1f}s)")
        if cm is not None:
            print("[VAL] Confusion Matrix:\n", cm)
            print("[VAL] Classification Report:\n", report)

        torch.save({"model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "embed_dim": 128,
                    "img_size": cfg.img_size}, os.path.join(cfg.out_dir, "model.pth"))
        if cfg.save_every and (epoch % cfg.save_every == 0):
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, f"model_e{epoch}.pth"))
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_best_only_weights.pth"))
    print(f"[DONE] Saved artifacts -> {cfg.out_dir}")

# ----------------------------
# Enrollment (gallery)
# ----------------------------

@torch.no_grad()
def extract_embedding(model: 'FaceEmbedder', img: Image.Image, tf, device: str):
    x = tf(img).unsqueeze(0).to(device)
    emb, _ = model(x)
    return emb.squeeze(0).cpu().numpy()

def load_model(model_path: str, device: str) -> 'FaceEmbedder':
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        num_classes = ckpt.get("num_classes", 1)
        embed_dim   = ckpt.get("embed_dim", 128)
        model = FaceEmbedder(num_classes=num_classes, embed_dim=embed_dim, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        return model.to(device).eval()
    else:
        model = FaceEmbedder(num_classes=1, embed_dim=128, pretrained=False)
        model.load_state_dict(ckpt)
        return model.to(device).eval()

def enroll_cmd(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(args.class_map):
        print(f"[ERR] class map not found: {args.class_map}", file=sys.stderr); sys.exit(2)
    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    model = load_model(args.model, device)
    _, eval_tf = make_transforms(img_size=args.img_size)

    people = sorted([d for d in os.listdir(args.gallery_root) if os.path.isdir(os.path.join(args.gallery_root, d))])
    if not people:
        print(f"[ERR] No subfolders found under: {args.gallery_root}", file=sys.stderr); sys.exit(3)
    print(f"[INFO] Enrolling: {people}")

    gallery: Dict[str, np.ndarray] = {}
    for person in people:
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            imgs.extend(glob.glob(os.path.join(args.gallery_root, person, ext)))
        if not imgs:
            print(f"[WARN] No images for {person}, skipping"); continue
        embs = []
        for p in imgs:
            try:
                img = Image.open(p).convert("RGB")
                emb = extract_embedding(model, img, eval_tf, device)
                embs.append(emb)
            except Exception as e:
                print(f"[WARN] {p}: {e}")
        if embs:
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
            gallery[person] = mean_emb
            print(f"[OK] {person}: {len(embs)} images")
    safe_makedirs(os.path.dirname(args.out) or ".")
    with open(args.out, "wb") as f:
        pickle.dump(gallery, f)
    print(f"[DONE] Gallery -> {args.out} (ids={len(gallery)})")

# ----------------------------
# Recognition
# ----------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def recognize_cmd(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    _, eval_tf = make_transforms(img_size=args.img_size)
    with open(args.gallery, "rb") as f:
        gallery: Dict[str, np.ndarray] = pickle.load(f)
    if not gallery:
        print("[ERR] Empty gallery", file=sys.stderr); sys.exit(2)

    img = Image.open(args.image).convert("RGB")
    emb = extract_embedding(model, img, eval_tf, device)
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    best_name, best_score = None, -1.0
    for name, mean_emb in gallery.items():
        s = cosine_sim(emb, mean_emb)
        if s > best_score:
            best_score = s
            best_name = name

    print(json.dumps({
        "predicted_identity": best_name,
        "cosine_similarity": round(best_score, 4),
        "threshold_hint": "Tune acceptance threshold on your validation set (~0.5–0.7 typical)."
    }, indent=2))

# ----------------------------
# Auto: Kaggle + Split + Train
# ----------------------------

def auto_kaggle_train_cmd(args):
    seed_all(args.seed)
    workspace = os.path.abspath(args.workspace)
    safe_makedirs(workspace)

    # Download
    dl_dir = os.path.join(workspace, "kaggle_raw")
    if not os.path.isdir(dl_dir) or not os.listdir(dl_dir):
        kaggle_download(args.kaggle_dataset, dl_dir)
    else:
        print(f"[INFO] Using existing download at {dl_dir}")

    # Detect identity folder
    id_root = detect_id_folder(dl_dir)
    if id_root is None:
        # Fallback for LFW common layout names
        candidates = [
            os.path.join(dl_dir, "lfw-deepfunneled"),
            os.path.join(dl_dir, "lfw"),
            os.path.join(dl_dir, "lfw_funneled"),
        ]
        id_root = next((c for c in candidates if os.path.isdir(c)), None)
    if id_root is None:
        print("[ERR] Could not detect identity folder with subfolders per person.", file=sys.stderr)
        sys.exit(5)
    print(f"[INFO] Identity folder: {id_root}")

    # Build splits
    data_root = os.path.join(workspace, "data_splits")
    if not (os.path.isdir(os.path.join(data_root, "train")) and os.path.isdir(os.path.join(data_root, "val"))):
        stratified_split_id_folder(id_root, data_root, val_ratio=args.val_ratio, min_images_per_id=args.min_per_id, seed=args.seed)
    else:
        print(f"[INFO] Using existing splits at {data_root}")

    # Train
    train_args = argparse.Namespace(
        data_root=data_root,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        workers=args.workers,
        seed=args.seed,
        save_every=args.save_every,
    )
    train_cmd(train_args)

# ----------------------------
# CLI
# ----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="CNN Face Recognition with Kaggle training")
    sub = p.add_subparsers(dest="cmd", required=True)

    # auto: download Kaggle -> split -> train
    pa = sub.add_parser("auto", help="Download a Kaggle dataset (default LFW), build train/val, and train.")
    pa.add_argument("--kaggle-dataset", default="jessicali9530/lfw-dataset", help="Kaggle dataset slug (owner/dataset)")
    pa.add_argument("--workspace", default="./workspace", help="Working directory for downloads/splits")
    pa.add_argument("--val-ratio", type=float, default=0.2)
    pa.add_argument("--min-per-id", type=int, default=2)
    pa.add_argument("--out", required=True, help="Output directory for artifacts")
    pa.add_argument("--epochs", type=int, default=10)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--weight-decay", type=float, default=1e-4)
    pa.add_argument("--img-size", type=int, default=224)
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--save-every", type=int, default=0)
    pa.set_defaults(func=auto_kaggle_train_cmd)

    # train (assumes you already have data_root/train and data_root/val)
    pt = sub.add_parser("train", help="Train on an existing ImageFolder split.")
    pt.add_argument("--data-root", required=True, help="Root containing train/ and val/")
    pt.add_argument("--out", required=True, help="Output directory for artifacts")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--img-size", type=int, default=224)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--save-every", type=int, default=0)
    pt.set_defaults(func=train_cmd)

    # enroll
    pe = sub.add_parser("enroll", help="Build an embedding gallery (mean embeddings per identity).")
    pe.add_argument("--gallery-root", required=True, help="Folder with subfolders per person")
    pe.add_argument("--model", required=True, help="Path to trained model .pth")
    pe.add_argument("--class-map", required=True, help="class_to_idx.json path")
    pe.add_argument("--out", required=True, help="Output gallery pickle path")
    pe.add_argument("--img-size", type=int, default=224)
    pe.set_defaults(func=enroll_cmd)

    # recognize
    pr = sub.add_parser("recognize", help="Recognize a face against the enrolled gallery.")
    pr.add_argument("--image", required=True, help="Path to input face image")
    pr.add_argument("--model", required=True, help="Path to trained model .pth")
    pr.add_argument("--class-map", required=True, help="class_to_idx.json path")
    pr.add_argument("--gallery", required=True, help="Gallery pickle path")
    pr.add_argument("--img-size", type=int, default=224)
    pr.set_defaults(func=recognize_cmd)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
