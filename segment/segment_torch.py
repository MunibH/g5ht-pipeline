import os
import sys
import torch
import numpy as np
import tifffile as tif
from scipy.stats import zscore
import segmentation_models_pytorch as smp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4             
CHECKPOINT = '/home/munib/CODE/g5ht-pipeline/segment/deeplab_v3p_resnet50_20251109.pth'   # from training step, train_torch.py

# ----------------------------------------
# Preprocessing
# ----------------------------------------
def preprocess_slice(rfp_slice):
    """Input: (H, W) float32 or (C,H,W) slice. Output: torch tensor [1,3,H,W]."""
    if rfp_slice.ndim == 3:
        rfp_slice = rfp_slice.max(axis=0)

    rfp_z = zscore(rfp_slice.astype(np.float32), axis=None)
    if np.isnan(rfp_z).any():
        rfp_z = np.nan_to_num(rfp_z, nan=0.0)

    img = np.stack([rfp_z, rfp_z, rfp_z], axis=0)  # CHW
    return torch.from_numpy(img).unsqueeze(0).float()  # [1,3,H,W]


# ----------------------------------------
# Model loading
# ----------------------------------------
def load_model():
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=2
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("model loaded")
    return model


# ----------------------------------------
# Batched inference
# ----------------------------------------
def predict_batch(model, imgs):
    """
    imgs: torch tensor [B,3,H,W]
    return: numpy array [B,H,W] bool
    """
    imgs = imgs.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        logits = model(imgs)             # [B,2,H,W]
        labels = torch.argmax(logits, dim=1)  # [B,H,W]
    return labels.cpu().numpy().astype(bool)


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == '__main__':
    
    mip_path = sys.argv[1]
    mip = tif.imread(os.path.join(mip_path,'mip.tif'))      # expected shape (Z, C, H, W) or (Z, H, W)
    name = os.path.splitext(os.path.basename(mip_path))[0]

    model = load_model()

    Z = len(mip)
    output = np.zeros((Z, 512, 512), dtype=bool)

    out_dir = mip_path
    out_path = os.path.join(out_dir, "label.tif")
    
    # ----------------------------------------
    # Iterate over batches
    # ----------------------------------------
    for i in range(0, Z, BATCH_SIZE):
        batch_indices = range(i, min(i + BATCH_SIZE, Z))

        batch_imgs = []
        for idx in batch_indices:
            slice_2d = mip[idx, 1] if mip.ndim == 4 else mip[idx]
            batch_imgs.append(preprocess_slice(slice_2d))

        batch_tensor = torch.cat(batch_imgs, dim=0)   # [B,3,H,W]
        batch_pred = predict_batch(model, batch_tensor)

        output[list(batch_indices)] = batch_pred
        print(f"{name}: {i}")

    print("OUTPUT: ", out_path)
    tif.imwrite(out_path, output)
