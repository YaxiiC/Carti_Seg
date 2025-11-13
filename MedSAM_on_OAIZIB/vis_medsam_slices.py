import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------ CONFIG ------------------
# Folder containing the prepared slices
IMG_DIR = r"C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM\imgs"
GT_DIR  = r"C:\Users\chris\MICCAI2026\MRI_Knee_OAIZIBCM\gts"

# Optional: manually specify a slice filename (without extension)
# Example: SLICE_NAME = "MRI_Knee_OAIZIBCM_oaizib_001-020"
SLICE_NAME = None  # set to None for random pick

# Define colors for label IDs (0–5 for OAIZIB-CM)
LABEL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),     # 0 background (transparent)
    (1.0, 0.0, 0.0, 0.5),     # 1 Femur (red)
    (0.0, 1.0, 0.0, 0.5),     # 2 Femoral Cartilage (green)
    (0.0, 0.0, 1.0, 0.5),     # 3 Tibia (blue)
    (1.0, 1.0, 0.0, 0.5),     # 4 Medial Tibial Cartilage (yellow)
    (1.0, 0.0, 1.0, 0.5),     # 5 Lateral Tibial Cartilage (magenta)
]
cmap = ListedColormap(LABEL_COLORS)

# ------------------ FUNCTIONS ------------------
def load_random_slice():
    imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(".npy")]
    if not imgs:
        raise FileNotFoundError(f"No .npy found in {IMG_DIR}")
    chosen = SLICE_NAME or random.choice(imgs).replace(".npy", "")
    img_path = os.path.join(IMG_DIR, chosen + ".npy")
    gt_path  = os.path.join(GT_DIR,  chosen + ".npy")
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing GT for {chosen}")
    img = np.load(img_path)
    gt = np.load(gt_path) if os.path.exists(gt_path) else np.zeros(img.shape[:2])
    return chosen, img, gt

def show_slice(img, gt, title=""):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[1].imshow(gt, cmap=cmap, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[1].set_title("Mask")
    axs[2].imshow(img)
    axs[2].imshow(gt, cmap=cmap, vmin=0, vmax=len(LABEL_COLORS)-1)
    axs[2].set_title("Overlay")
    for a in axs:
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    case, img, gt = load_random_slice()
    print(f"Visualizing {case}")
    show_slice(img, gt, title=case)
