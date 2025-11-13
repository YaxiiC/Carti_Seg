# -*- coding: utf-8 -*-
#python prep_oaizibcm_for_medsam.py ^
#  --dataset_root "C:\Users\chris\MICCAI2026\OAI-ZIB-CM" ^
#  --out_dir "C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM" ^
#  --process_test

"""
Prepare OAI-ZIB-CM dataset (image: oaizib_<case>_0000.nii, label: oaizib_<case>.nii)
for MedSAM training.
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from skimage import transform
from tqdm import tqdm
import cc3d

join = os.path.join

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def list_oai_cases(img_dir: str, lab_dir: str) -> list:
    """
    Match:
      image: oaizib_<case>_0000.nii  OR oaizib_<case>_0000.nii.gz
      label: oaizib_<case>.nii       OR oaizib_<case>.nii.gz
    Returns a sorted list of <case> strings (e.g., 'oaizib_001').
    """
    if not os.path.isdir(img_dir) or not os.path.isdir(lab_dir):
        print(f"[WARN] Missing folder(s): img_dir={img_dir}, lab_dir={lab_dir}")
        return []

    files = os.listdir(img_dir)
    cases = []
    for f in files:
        if not (f.endswith("_0000.nii") or f.endswith("_0000.nii.gz")):
            continue
        # strip the _0000.nii or _0000.nii.gz
        case = f.replace("_0000.nii.gz", "").replace("_0000.nii", "")
        # label could be .nii or .nii.gz
        lab_nii     = os.path.join(lab_dir, f"{case}.nii")
        lab_niigz   = os.path.join(lab_dir, f"{case}.nii.gz")
        if os.path.exists(lab_nii) or os.path.exists(lab_niigz):
            cases.append(case)
    cases = sorted(set(cases))

    if len(cases) == 0:
        print("[DEBUG] No matches. Sample files I see in images dir:")
        for f in sorted(files)[:10]:
            print("   ", f)
        lab_files = os.listdir(lab_dir)
        print("[DEBUG] Sample files I see in labels dir:")
        for f in sorted(lab_files)[:10]:
            print("   ", f)

    return cases


def robust_mri_normalize(vol: np.ndarray, lo=0.5, hi=99.5) -> np.uint8:
    nz = vol[vol > 0]
    if nz.size < 10:
        mn, mx = np.min(vol), np.max(vol)
    else:
        mn, mx = np.percentile(nz, lo), np.percentile(nz, hi)
    v = np.clip(vol, mn, mx)
    v = (v - v.min()) / max(v.max() - v.min(), 1e-6)
    v = (v * 255.0).astype(np.uint8)
    v[vol == 0] = 0
    return v

def remove_small_3d_and_2d(lab, vox3d_thresh=1000, vox2d_thresh=100):
    lab = cc3d.dust(lab, threshold=vox3d_thresh, connectivity=26, in_place=False)
    for z in range(lab.shape[0]):
        lab[z] = cc3d.dust(lab[z], threshold=vox2d_thresh, connectivity=8, in_place=False)
    return lab

def find_nonzero_slices(lab):
    return np.unique(np.where(lab > 0)[0])

def save_sitk_like(vol_np, ref_img, out_path):
    out = sitk.GetImageFromArray(vol_np)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, out_path)

def resize_img_mask_2d(img2d_u8, mask2d_u8, out_size=1024):
    img3 = np.repeat(img2d_u8[..., None], 3, axis=-1)
    img_resized = transform.resize(
        img3, (out_size, out_size),
        order=3, preserve_range=True, mode="constant", anti_aliasing=True
    ).astype(np.float32)
    mn, mx = img_resized.min(), img_resized.max()
    img_resized = (img_resized - mn) / max(mx - mn, 1e-8)
    mask_resized = transform.resize(
        mask2d_u8, (out_size, out_size),
        order=0, preserve_range=True, mode="constant", anti_aliasing=False
    ).astype(np.uint8)
    return img_resized, mask_resized

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root containing imagesTr/labelsTr and optionally imagesTs/labelsTs")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="MRI_Knee_OAIZIBCM_")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--vox3d_thresh", type=int, default=1000)
    parser.add_argument("--vox2d_thresh", type=int, default=100)
    parser.add_argument("--process_test", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tr_img_dir = join(args.dataset_root, "imagesTr")
    tr_lab_dir = join(args.dataset_root, "labelsTr")
    ts_img_dir = join(args.dataset_root, "imagesTs")
    ts_lab_dir = join(args.dataset_root, "labelsTs")

    ensure_dirs(args.out_dir)
    ensure_dirs(join(args.out_dir, "imgs"))
    ensure_dirs(join(args.out_dir, "gts"))

    train_cases = list_oai_cases(tr_img_dir, tr_lab_dir)
    if args.limit:
        train_cases = train_cases[:args.limit]
    print(f"[Train] {len(train_cases)} cases found.")

    test_cases = []
    if args.process_test and os.path.isdir(ts_img_dir):
        test_cases = list_oai_cases(ts_img_dir, ts_lab_dir)
        if args.limit:
            test_cases = test_cases[:args.limit]
        print(f"[Test] {len(test_cases)} cases found.")

    def process_split(cases, img_dir, lab_dir, split):
        for case in tqdm(cases, desc=f"{split}"):
            img_path = join(img_dir, f"{case}_0000.nii")
            lab_path = join(lab_dir, f"{case}.nii")
            if not os.path.exists(img_path) or not os.path.exists(lab_path):
                print(f"Skip {case}: missing file.")
                continue

            lab_img = sitk.ReadImage(lab_path)
            lab_np = sitk.GetArrayFromImage(lab_img).astype(np.uint8)
            lab_np = remove_small_3d_and_2d(lab_np, args.vox3d_thresh, args.vox2d_thresh)
            z_idx = find_nonzero_slices(lab_np)
            if len(z_idx) == 0:
                print(f"Skip {case}: empty mask.")
                continue
            zmin, zmax = z_idx.min(), z_idx.max() + 1
            lab_roi = lab_np[zmin:zmax]

            img_itk = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img_itk)
            img_u8 = robust_mri_normalize(img_np)
            img_roi = img_u8[zmin:zmax]

            # save per-case npz and sanity-check nii
            np.savez_compressed(join(args.out_dir, f"{args.prefix}{case}.npz"),
                                imgs=img_roi, gts=lab_roi, spacing=img_itk.GetSpacing())
            save_sitk_like(img_roi, img_itk, join(args.out_dir, f"{args.prefix}{case}_img.nii.gz"))
            save_sitk_like(lab_roi, img_itk, join(args.out_dir, f"{args.prefix}{case}_gt.nii.gz"))

            # per-slice npy
            for i in range(img_roi.shape[0]):
                img2d, msk2d = img_roi[i], lab_roi[i]
                img_r, msk_r = resize_img_mask_2d(img2d, msk2d, args.image_size)
                base = f"{args.prefix}{case}-{str(i).zfill(3)}"
                np.save(join(args.out_dir, "imgs", f"{base}.npy"), img_r)
                np.save(join(args.out_dir, "gts", f"{base}.npy"),  msk_r)

    process_split(train_cases, tr_img_dir, tr_lab_dir, "train")
    if test_cases:
        process_split(test_cases, ts_img_dir, ts_lab_dir, "test")

    print("✅ Done preparing OAI-ZIB-CM for MedSAM.")

if __name__ == "__main__":
    main()
