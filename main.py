import os
import glob
import argparse
import yaml
import pickle
import torch
import numpy as np
import SimpleITK as sitk

from utils import compute_exact_sdf, extract_boundary_samples, interpolate_curve
from train import train_base_sdf, train_deformation, generate_deformed_mask


DEFAULT_SDF_TRAIN_CONFIG = {
    'epochs': 500,
    'batch_size': 262144, # 2^18
    'lr': 0.001,
    'trunc_margin': 5.0,
}

DEFAULT_DEFORM_TRAIN_CONFIG = {
    'epochs': 1000,
    'lr': 0.001,
    'alpha': 1.0,   # Curve Loss
    'beta': 0.1,      # Smoothness Loss
    'gamma': 0.01,      # Anchor Loss
    'lambda': 2.0,
    'sigma_p': 0.02,
    'sigma_t_scale': 0.5, # delta_t * 0.5
    'batch_size_s': 16384, # 2^14
    'batch_size_a': 16384, # 2^14
}

CACHE_DIR = "./model_cache"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data_and_volume_info(cfg: dict):
    pred_files = sorted(glob.glob(os.path.join(cfg['data']['input_dir'], "*.mha")))
    assert len(pred_files) > 0, f"No MHA files found in {cfg['data']['input_dir']}"

    target_id = cfg['data']['target_id']
    
    mask_list = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in pred_files]
    mask4d = np.stack(mask_list, axis=0) # [T, Z, Y, X]
    T, Z, Y, X = mask4d.shape

    target_mask = (mask4d == target_id)
    if np.any(target_mask):
        z_any = np.any(target_mask, axis=(0, 2, 3))
        y_any = np.any(target_mask, axis=(0, 1, 3))
        x_any = np.any(target_mask, axis=(0, 1, 2))
        
        z_min, z_max = np.where(z_any)[0][[0, -1]]
        y_min, y_max = np.where(y_any)[0][[0, -1]]
        x_min, x_max = np.where(x_any)[0][[0, -1]]
        
        margin = 20
        z_min, z_max = max(0, z_min - margin), min(Z, z_max + margin + 1)
        y_min, y_max = max(0, y_min - margin), min(Y, y_max + margin + 1)
        x_min, x_max = max(0, x_min - margin), min(X, x_max + margin + 1)
    else:
        z_min, z_max, y_min, y_max, x_min, x_max = 0, Z, 0, Y, 0, X

    cropped_mask4d = mask4d[:, z_min:z_max, y_min:y_max, x_min:x_max]
    
    sitk_img0 = sitk.ReadImage(pred_files[0])
    spacing = sitk_img0.GetSpacing() # (x, y, z)
    voxel_spacing = (spacing[2], spacing[1], spacing[0]) # (z, y, x)

    volume_info = {
        "spacing": voxel_spacing,
        "original_shape": (T, Z, Y, X),
        "cropped_shape": cropped_mask4d.shape,
        "offset": (z_min, y_min, x_min), # (z, y, x)
        "target_id": target_id,
        "ref_file": pred_files[0]
    }

    print("[*] Computing Exact 4D Distance Fields...")
    sdf_list = []
    for t in range(T):
        frame_mask = (cropped_mask4d[t] == target_id)
        sdf_3d = compute_exact_sdf(frame_mask, voxel_spacing)
        sdf_list.append(sdf_3d)
        
    sdf_volume_4d = np.stack(sdf_list, axis=0)
    return sdf_volume_4d, volume_info, pred_files


def get_or_train_base_sdf_cached(cfg: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    norm_path = os.path.normpath(cfg['data']['input_dir'])
    parts = norm_path.split(os.sep)
    input_dir_name = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]
    target_id = cfg['data']['target_id']
    cache_prefix = os.path.join(CACHE_DIR, f"{input_dir_name}_target{target_id}")
    
    model_path = f"{cache_prefix}_base_model.pth"
    info_path = f"{cache_prefix}_volume_info.pkl"
    sdf_path = f"{cache_prefix}_sdf_volume.npz"

    if os.path.exists(model_path) and os.path.exists(info_path) and os.path.exists(sdf_path):
        print(f"[*] Found cached base model & SDF data for '{input_dir_name}' (target ID: {target_id}). Loading...")
        
        with open(info_path, "rb") as f:
            volume_info = pickle.load(f)
            
        sdf_volume_4d = np.load(sdf_path)['sdf']
        base_sdf_model = torch.load(model_path)
        
        pred_files = sorted(glob.glob(os.path.join(cfg['data']['input_dir'], "*.mha")))
        return base_sdf_model, sdf_volume_4d, volume_info, pred_files

    print(f"[*] No cache found for '{input_dir_name}' (target ID: {target_id}). Starting preparation & pre-training...")
    sdf_volume_4d, volume_info, pred_files = prepare_data_and_volume_info(cfg)
    base_sdf_model = train_base_sdf(sdf_volume_4d, DEFAULT_SDF_TRAIN_CONFIG)

    torch.save(base_sdf_model, model_path)
    with open(info_path, "wb") as f:
        pickle.dump(volume_info, f)
    np.savez_compressed(sdf_path, sdf=sdf_volume_4d)

    print(f"[+] Successfully saved cache to '{cache_prefix}_*'")
    return base_sdf_model, sdf_volume_4d, volume_info, pred_files


def parse_control_points(cfg: dict, volume_info: dict, sdf_volume_4d: np.ndarray) -> np.ndarray:
    curves_list = cfg.get('curves', None)
    
    T_orig = volume_info['original_shape'][0]
    T_crop = volume_info['cropped_shape'][0]
    z_min, y_min, x_min = volume_info['offset']

    ref_img = sitk.ReadImage(volume_info['ref_file'])

    curve_points_4d = []
    edited_frames = set()

    if curves_list and len(curves_list) > 0:
        print(f"[*] Loaded {len(curves_list)} curves from config.")
        
        for c_idx, raw_cps in enumerate(curves_list):
            cps_np = np.array(raw_cps, dtype=np.float32)  # [N, 4] -> (phys_x, phys_y, phys_z, frame_idx)
            if len(cps_np) == 0:
                continue

            f_idx = int(round(cps_np[0, 3]))
            t_val = f_idx / float(max(1, T_orig - 1))
            edited_frames.add(f_idx)

            cropped_vox_cps = []
            for row in cps_np:
                phys_pt = (float(row[0]), float(row[1]), float(row[2]))
                vox_x, vox_y, vox_z = ref_img.TransformPhysicalPointToContinuousIndex(phys_pt)
                
                crop_x = vox_x - x_min
                crop_y = vox_y - y_min
                crop_z = vox_z - z_min
                cropped_vox_cps.append([crop_x, crop_y, crop_z])

            cropped_vox_cps = np.array(cropped_vox_cps, dtype=np.float32)

            if len(cropped_vox_cps) >= 2:
                dense_xyz = interpolate_curve(cropped_vox_cps, num_points=100)
            else:
                dense_xyz = cropped_vox_cps

            for pt in dense_xyz:
                curve_points_4d.append([pt[0], pt[1], pt[2], t_val])

    for f_idx in range(T_crop):
        if f_idx not in edited_frames:
            t_val = f_idx / float(max(1, T_crop - 1))
            frame_mask = (sdf_volume_4d[f_idx] <= 0)
            boundary_pts = extract_boundary_samples(frame_mask, num_samples=150)
            for pt in boundary_pts:
                curve_points_4d.append([pt[0], pt[1], pt[2], t_val])

    print(f"[*] Total constraint points: {len(curve_points_4d)} (Edited frames: {len(edited_frames)}/{T_crop})")
    return np.array(curve_points_4d, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Spatiotemporal Deformation framework for 4D Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_sdf_model, sdf_volume_4d, volume_info, ref_pred_files = get_or_train_base_sdf_cached(cfg)

    pts_curve_4d = parse_control_points(cfg, volume_info, sdf_volume_4d)

    deform_model = train_deformation(
        base_model=base_sdf_model,
        pts_curve_4d=pts_curve_4d,
        crop_shape=volume_info['cropped_shape'],
        config=DEFAULT_DEFORM_TRAIN_CONFIG
    )

    out_frames = cfg.get('volume', {}).get('num_output_frames', None)
    deformed_mask = generate_deformed_mask(
        base_model=base_sdf_model,
        deform_model=deform_model,
        volume_info=volume_info,
        out_frames=out_frames
    )
    
    out_dir = cfg['data']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    num_out = deformed_mask.shape[0]
    for fi in range(num_out):
        out_path = os.path.join(out_dir, f"deformed_{fi:02d}.mha")
        out_img = sitk.GetImageFromArray(deformed_mask[fi])
        
        ref_idx = min(fi, len(ref_pred_files) - 1)
        ref_sitk = sitk.ReadImage(ref_pred_files[ref_idx])
        out_img.CopyInformation(ref_sitk)
        
        sitk.WriteImage(out_img, out_path)

    print(f"[+] Successfully saved deformed MHA frames to '{out_dir}'")


if __name__ == "__main__":
    main()