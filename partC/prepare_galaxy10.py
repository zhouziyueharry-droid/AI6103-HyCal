"""Fast, output-compatible replacement for the official Galaxy10 DECaLS converter.

third_party/HyCal-CIL/task_datasets/preprocess_galaxy10_decals.py reads the HDF5 one image at a
time; the file is gzip-chunked 555 images per chunk, so that takes ~1 image/s (about 5 hours).
This script does the same thing but reads whole 555-image chunks:
  * same stratified 80/10/10 split (sklearn train_test_split, random_state=42, two stages)
  * same file names ({idx:05d}.jpg), same PIL default JPEG encoding
  * same CSV (split, filename, class_num) in dataset order
Output goes to the location the official loader expects:
  data/Galaxy10_DECals/_galaxy_temp/{images/, galaxy10_split.csv}
"""
import csv
import os

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import common as C

H5 = os.path.join(C.DATA_ROOT, "Galaxy10_DECals", "Galaxy10_DECals.h5")


def main():
    if os.path.lexists(C.GALAXY_DIR):
        raise FileExistsError(f"Output already exists: {C.GALAXY_DIR}")
    os.makedirs(C.GALAXY_IMAGES)

    with h5py.File(H5, "r") as f:
        labels = np.array(f["ans"])
        n = f["images"].shape[0]
        indices = np.arange(n)
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)
        split = np.empty(n, dtype=object)
        split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"

        step = f["images"].chunks[0]
        for start in range(0, n, step):
            block = f["images"][start:start + step]
            for j, arr in enumerate(block):
                Image.fromarray(arr).save(os.path.join(C.GALAXY_IMAGES, f"{start + j:05d}.jpg"))
            print(f"{min(start + step, n)}/{n}", flush=True)

    with open(C.GALAXY_SPLIT, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["split", "filename", "class_num"])
        for i in range(n):
            w.writerow([split[i], f"{i:05d}.jpg", int(labels[i])])
    print("saved", C.GALAXY_DIR)


if __name__ == "__main__":
    main()
