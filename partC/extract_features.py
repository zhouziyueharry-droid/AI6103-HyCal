"""Extract frozen CLIP features once per dataset and cache them.

Each image is encoded exactly once, in the order of the fixed split file, using the official
HyCal preprocessing (Resize(224, bicubic) -> CenterCrop(224) -> normalize) and the vendored
OpenAI CLIP.

Output (partC/cache/{dataset}_ViT-B-16.npz):
  {train,val,test}_feats   float32 [N, 512], L2-normalised image embeddings
  {train,val,test}_labels  int64   [N], dataset-local label
  {train,val,test}_paths   str     [N], path relative to the dataset image folder
  classnames               str     [C], names used in text prompts (official wording)
  display_names            str     [C], short names for tables/plots
  class_ids                str     [C], global ids "{dataset}:{label}"
  text_feats               float32 [C, 512], L2-normalised prompt embeddings
  extract_seconds          float   wall time of image encoding (all splits)
  extract_peak_rss_mb      float   peak resident memory of this process
  extract_peak_mps_mb      float   peak MPS driver memory seen after a batch (0 if not on MPS)

Usage: python extract_features.py --dataset eurosat|galaxy10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import common as C

sys.path.insert(0, C.HYCAL_REPO)


class ImageList(torch.utils.data.Dataset):
    def __init__(self, root, paths, transform):
        self.root, self.paths, self.transform = root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.root, self.paths[i])).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eurosat", choices=C.DATASETS)
    args = ap.parse_args()
    import clip  # vendored OpenAI CLIP from the shared upstream directory
    ds = C.load_dataset(args.dataset)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    # Load on CPU so weights stay float32 (same as the official CPU path), then move.
    model, _, preprocess = clip.load(C.CLIP_MODEL, device="cpu", jit=False)
    model = model.float().eval().to(device)
    print(f"device={device}; preprocess={preprocess}")

    out, total, peak_mps = {}, 0.0, 0.0
    with torch.no_grad():
        for name in ["train", "val", "test"]:
            paths = [p for p, _ in ds["splits"][name]]
            labels = np.array([l for _, l in ds["splits"][name]], dtype=np.int64)
            loader = torch.utils.data.DataLoader(ImageList(ds["image_root"], paths, preprocess),
                                                 batch_size=64, num_workers=0)
            feats, t0 = [], time.time()
            for x in loader:
                f = model.encode_image(x.to(device)).float()
                feats.append((f / (f.norm(dim=-1, keepdim=True) + 1e-10)).cpu())
                if device == "mps":
                    peak_mps = max(peak_mps, torch.mps.driver_allocated_memory() / 1024 ** 2)
            total += time.time() - t0
            out[f"{name}_feats"] = torch.cat(feats).numpy().astype(np.float32)
            out[f"{name}_labels"] = labels
            out[f"{name}_paths"] = np.array(paths)
            print(f"{name}: {len(paths)} images in {time.time() - t0:.1f}s")

        tokens = clip.tokenize([C.TEMPLATE.format(n.replace("_", " ")) for n in ds["prompt_names"]]).to(device)
        t = model.encode_text(tokens).float()
        out["text_feats"] = (t / (t.norm(dim=-1, keepdim=True) + 1e-10)).cpu().numpy().astype(np.float32)
    out["classnames"] = np.array(ds["prompt_names"])
    out["display_names"] = np.array(ds["display_names"])
    out["class_ids"] = np.array([f"{args.dataset}:{i}" for i in range(len(ds["prompt_names"]))])
    out["extract_seconds"] = np.array(total)
    peak_rss = C.peak_rss_mb()
    if peak_rss is not None:
        out["extract_peak_rss_mb"] = np.array(peak_rss)
    out["extract_peak_mps_mb"] = np.array(peak_mps)

    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez(C.feature_cache(args.dataset), **out)
    print("saved", C.feature_cache(args.dataset), {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
