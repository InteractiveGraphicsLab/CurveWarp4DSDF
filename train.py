import time
import numpy as np
import torch
import torch.nn as nn
from models import NeuralSDF_4D, DeformerMLP


def train_base_sdf(
    sdf_volume_4d: np.ndarray,
    config: dict
) -> NeuralSDF_4D:
    sdf_tensor = torch.tensor(sdf_volume_4d, dtype=torch.float32, device="cuda")
    T, Z, Y, X = sdf_tensor.shape

    trunc_margin = config['trunc_margin']
    sdf_tensor = torch.clamp(sdf_tensor, -trunc_margin, trunc_margin) / trunc_margin

    model = NeuralSDF_4D().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()

    surface_indices = torch.nonzero(torch.abs(sdf_tensor) < 0.4)
    all_indices = torch.nonzero(torch.ones_like(sdf_tensor))

    epochs = config['epochs']
    batch_size = config['batch_size']

    print(f"[*] Training Base 4D Neural SDF ({epochs} epochs)...")
    start = time.time()

    for epoch in range(epochs):
        n_surface = int(batch_size * 0.7)
        n_uniform = batch_size - n_surface

        idx_surf = torch.randint(0, len(surface_indices), (n_surface,), device="cuda")
        idx_uni = torch.randint(0, len(all_indices), (n_uniform,), device="cuda")
        sampled = torch.cat([surface_indices[idx_surf], all_indices[idx_uni]], dim=0)

        t_idx, z_idx, y_idx, x_idx = sampled[:, 0], sampled[:, 1], sampled[:, 2], sampled[:, 3]
        
        pts = torch.stack([
            x_idx.float() / max(1.0, X - 1.0),
            y_idx.float() / max(1.0, Y - 1.0),
            z_idx.float() / max(1.0, Z - 1.0),
            t_idx.float() / max(1.0, T - 1.0)
        ], dim=-1)

        target_sdf = sdf_tensor[t_idx, z_idx, y_idx, x_idx].unsqueeze(-1)

        optimizer.zero_grad()
        pred_sdf = model(pts).float()
        loss = loss_fn(pred_sdf, target_sdf)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.6f}")

    print(f"[+] Base SDF Training complete ({time.time() - start:.2f}s)")
    return model


def train_deformation(
    base_model: NeuralSDF_4D,
    pts_curve_4d: np.ndarray,
    crop_shape: tuple,
    config: dict
) -> DeformerMLP:
    T, Z, Y, X = crop_shape
    deform_model = DeformerMLP().cuda()

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    if len(pts_curve_4d) == 0:
        print("[!] Warning: No curve constraints provided. Returning zero-initialized deformer.")
        return deform_model

    pts_curve_tensor = torch.tensor(pts_curve_4d, device="cuda", dtype=torch.float32)
    pts_curve_tensor[:, 0] /= max(1.0, X - 1.0)
    pts_curve_tensor[:, 1] /= max(1.0, Y - 1.0)
    pts_curve_tensor[:, 2] /= max(1.0, Z - 1.0)

    optimizer = torch.optim.Adam(deform_model.parameters(), lr=config['lr'])
    delta_t = 1.0 / max(1, T - 1)
    sigma_t = config['sigma_t_scale'] * delta_t

    epochs = config['epochs']
    print(f"[*] Training Displacement Field ({epochs} epochs)...")
    start = time.time()

    deform_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Curve Loss
        c_spatial = pts_curve_tensor[:, :3]
        t_curve = pts_curve_tensor[:, 3:4]

        pred_disp = deform_model(pts_curve_tensor)
        warped_curve_4d = torch.cat([c_spatial + pred_disp, t_curve], dim=-1)
        sdf_at_curve = base_model(warped_curve_4d)
        loss_curve = torch.mean(sdf_at_curve ** 2)

        # 2. Smoothness Loss
        pts_smooth = torch.rand((config['batch_size_s'], 4), device="cuda")
        eps_p = torch.randn((config['batch_size_s'], 3), device="cuda") * config['sigma_p']
        eps_t = torch.randn((config['batch_size_s'], 1), device="cuda") * sigma_t
        eps = torch.cat([eps_p, eps_t], dim=-1)

        norm_eps_sq = torch.sum(eps ** 2, dim=-1, keepdim=True) + 1e-8
        disp_orig = deform_model(pts_smooth)
        disp_pert = deform_model(pts_smooth + eps)
        loss_smooth = torch.mean(torch.sum((disp_orig - disp_pert) ** 2, dim=-1, keepdim=True) / norm_eps_sq)

        # 3. Anchor Loss
        pts_anchor = torch.rand((config['batch_size_a'], 4), device="cuda")
        disp_anchor = deform_model(pts_anchor)

        with torch.no_grad():
            p_anchor, t_anchor = pts_anchor[:, :3], pts_anchor[:, 3:4]
            p_curve, t_curve_ref = pts_curve_tensor[:, :3], pts_curve_tensor[:, 3:4]

            dist_p_sq = torch.cdist(p_anchor, p_curve, p=2) ** 2
            dist_t_sq = (t_anchor - t_curve_ref.T) ** 2
            d_4d = dist_p_sq + config['lambda'] * dist_t_sq
            min_d, _ = torch.min(d_4d, dim=1, keepdim=True)

        loss_anchor = torch.mean(min_d * torch.sum(disp_anchor ** 2, dim=-1, keepdim=True))

        total_loss = (
            config['alpha'] * loss_curve +
            config['beta'] * loss_smooth +
            config['gamma'] * loss_anchor
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(deform_model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 200 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:4d}/{epochs} | Loss: {total_loss.item():.5f} "
                  f"(Curve: {loss_curve.item():.4f}, Smooth: {loss_smooth.item():.4f}, Anchor: {loss_anchor.item():.4f})")

    print(f"[+] Deformation Training complete ({time.time() - start:.2f}s)")
    return deform_model


def generate_deformed_mask(
    base_model: NeuralSDF_4D,
    deform_model: DeformerMLP,
    volume_info: dict,
    out_frames: int = None
) -> np.ndarray:
    T_orig, Z_orig, Y_orig, X_orig = volume_info['original_shape']
    z_min, y_min, x_min = volume_info['offset']
    T_crop, Z, Y, X = volume_info['cropped_shape']
    target_id = volume_info['target_id']

    num_t = out_frames if out_frames is not None else T_orig

    print(f"[*] Generating Deformed Mask (Frames: {num_t})...")
    start = time.time()

    base_model.eval()
    deform_model.eval()

    full_mask = np.zeros((num_t, Z_orig, Y_orig, X_orig), dtype=np.uint8)

    grid_z = torch.linspace(0, 1, Z, device="cuda")
    grid_y = torch.linspace(0, 1, Y, device="cuda")
    grid_x = torch.linspace(0, 1, X, device="cuda")
    grid_x_m, grid_y_m, grid_z_m = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    
    eval_batch_size = 2**19

    with torch.no_grad():
        for t_idx in range(num_t):
            t_val = t_idx / float(max(1, num_t - 1))
            grid_t_m = torch.full_like(grid_x_m, fill_value=t_val)
            coords = torch.stack([grid_x_m, grid_y_m, grid_z_m, grid_t_m], dim=-1).reshape(-1, 4)

            sdf_values = []
            for i in range(0, coords.shape[0], eval_batch_size):
                batch_coords = coords[i : i + eval_batch_size]
                disp = deform_model(batch_coords).float()

                warped_coords = batch_coords.clone()
                warped_coords[:, :3] += disp

                out = base_model(warped_coords).float()
                sdf_values.append(out.cpu().numpy())

            sdf_volume = np.concatenate(sdf_values).reshape(X, Y, Z)
            
            cropped_mask = (sdf_volume <= 0.0).astype(np.uint8) * target_id
            cropped_mask = np.transpose(cropped_mask, (2, 1, 0)) # [Z, Y, X]
            
            full_mask[t_idx, z_min:z_min+Z, y_min:y_min+Y, x_min:x_min+X] = cropped_mask

    print(f"[+] Mask Generation complete ({time.time() - start:.2f}s)")
    return full_mask