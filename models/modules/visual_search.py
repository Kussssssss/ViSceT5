"""
models/modules/visual_search.py
VisualSearch — ConvNeXtV2-based attention crop.
"""

from typing import Optional, List, Dict, Any, Tuple
import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw
from matplotlib import cm
from transformers import AutoImageProcessor
from transformers import ConvNextV2Model

class VisualSearch(nn.Module):
    def __init__(self, vit_processor, model_config, vit_dim=768,
                 device=None, vs_local_dir=None, local_files_only=False):
        super().__init__()
        self.cfg = model_config
        self.vit_processor = vit_processor
        self.dim = int(vit_dim)

        backbone = str(getattr(self.cfg, "vs_backbone", "facebook/convnextv2-tiny-22k-224"))
        need_local = bool(local_files_only)

        cnn_ok = proc_ok = False
        if vs_local_dir and os.path.isdir(vs_local_dir):
            bb_dir  = os.path.join(vs_local_dir, "backbone")
            bbp_dir = os.path.join(vs_local_dir, "backbone_processor")

            if os.path.isdir(bb_dir):
                try:
                    self.cnn = ConvNextV2Model.from_pretrained(bb_dir, local_files_only=True)
                    cnn_ok = True
                except Exception as e:
                    self.cnn = None

            if os.path.isdir(bbp_dir):
                try:
                    self.processor = AutoImageProcessor.from_pretrained(bbp_dir, local_files_only=True)
                    proc_ok = True
                except Exception as e:
                    self.processor = None

        if need_local and (not cnn_ok or not proc_ok):
            missing = []
            if not cnn_ok:
                missing.append(f"backbone/ (expected at: {os.path.join(vs_local_dir or 'N/A', 'backbone')})")
            if not proc_ok:
                missing.append(f"backbone_processor/ (expected at: {os.path.join(vs_local_dir or 'N/A', 'backbone_processor')})")

            raise OSError(
                f"VisualSearch: Offline mode enabled but missing components:\n" +
                "\n".join(f"  - {m}" for m in missing) +
                f"\nvs_local_dir = {vs_local_dir}"
            )

        if not cnn_ok:
            if need_local:
                raise OSError("Offline/local-only but ConvNeXt backbone not found")
            self.cnn = ConvNextV2Model.from_pretrained(backbone, local_files_only=False)

        if not proc_ok:
            if need_local:
                raise OSError("Offline/local-only but ImageProcessor not found")
            self.processor = AutoImageProcessor.from_pretrained(backbone, local_files_only=False)

        init_image_size = self._pick_size(self.vit_processor)
        init_patch_size = int(getattr(self.cfg, "vs_patch_size", 16))

        vit_mean = getattr(self.vit_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        vit_std = getattr(self.vit_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.register_buffer("vit_mean", torch.tensor(vit_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("vit_std", torch.tensor(vit_std, dtype=torch.float32), persistent=False)

        cnn_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        cnn_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer("cnn_mean", torch.tensor(cnn_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("cnn_std", torch.tensor(cnn_std, dtype=torch.float32), persistent=False)

        cnn_input_size = self._pick_size(self.processor)
        self.register_buffer("cnn_input_size", torch.tensor(cnn_input_size, dtype=torch.long))

        self.register_buffer("attention_threshold", torch.tensor(0.7, dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("image_size", torch.tensor(init_image_size, dtype=torch.long))
        self.register_buffer("patch_size", torch.tensor(init_patch_size, dtype=torch.long))
        self.register_buffer("vs_percentile", torch.tensor(float(getattr(self.cfg, "vs_percentile", 99.0)), dtype=torch.float32))
        self.register_buffer("vs_dilate_half_patch", torch.tensor(bool(getattr(self.cfg, "vs_dilate_half_patch", True))))
        self.register_buffer("vs_min_area", torch.tensor(int(getattr(self.cfg, "vs_min_area", 1)), dtype=torch.long))
        self.register_buffer("vs_max_cov", torch.tensor(float(getattr(self.cfg, "vs_max_cov", 1.0)), dtype=torch.float32))
        self.register_buffer("vs_margin", torch.tensor(int(getattr(self.cfg, "vs_margin", 0)), dtype=torch.long))

        if vs_local_dir and os.path.isdir(vs_local_dir):
            params_path = os.path.join(vs_local_dir, "visual_search_params.pt")

            if os.path.isfile(params_path):
                try:
                    params = torch.load(params_path, map_location="cpu")

                    if "attention_threshold" in params:
                        self.attention_threshold.copy_(params["attention_threshold"])
                    if "temperature" in params:
                        self.temperature.copy_(params["temperature"])
                    if "image_size" in params:
                        self.image_size.fill_(int(params["image_size"]))
                    if "patch_size" in params:
                        self.patch_size.fill_(int(params["patch_size"]))
                    if "vs_percentile" in params:
                        self.vs_percentile.copy_(params["vs_percentile"])
                    if "vs_dilate_half_patch" in params:
                        self.vs_dilate_half_patch.copy_(params["vs_dilate_half_patch"])
                    if "vs_min_area" in params:
                        self.vs_min_area.fill_(int(params["vs_min_area"]))
                    if "vs_max_cov" in params:
                        self.vs_max_cov.copy_(params["vs_max_cov"])
                    if "vs_margin" in params:
                        self.vs_margin.fill_(int(params["vs_margin"]))
                    if "cnn_input_size" in params:
                        self.cnn_input_size.fill_(int(params["cnn_input_size"]))

                except Exception as e:
                    print(f"⚠ Error loading VS params: {e}")

    @staticmethod
    def _pick_size(p):
        if hasattr(p, "crop_size") and isinstance(p.crop_size, dict):
            return int(p.crop_size.get("width") or p.crop_size.get("height") or 224)
        if hasattr(p, "size") and isinstance(p.size, dict):
            if "shortest_edge" in p.size and not (hasattr(p, "do_center_crop") and p.do_center_crop):
                 return int(p.size["shortest_edge"])
            return int(p.size.get("width") or p.size.get("height") or 224)
        return int(p) if isinstance(p, (int, float)) else 224


    @staticmethod
    def _quantile(x, q):
        B, H, W = x.shape
        k = max(1, int(round((1.0 - q) * H * W)))
        vals, _ = torch.topk(x.reshape(B, -1), k, dim=1, largest=True, sorted=True)
        thr = vals[:, -1]
        return thr.view(B, 1, 1)

    @staticmethod
    def _dilate_mask(mask, ksize=3, iters=1):
        if iters <= 0 or ksize <= 1: return mask
        pad = ksize // 2
        m = mask.unsqueeze(1).float()
        for _ in range(iters):
            m = F.max_pool2d(m, kernel_size=ksize, stride=1, padding=pad)
        return (m.squeeze(1) > 0.5).to(mask.dtype)

    @staticmethod
    def _bbox_from_mask(mask):
        B, H, W = mask.shape
        boxes = []
        for b in range(B):
            ys, xs = torch.where(mask[b] > 0)
            if ys.numel() == 0:
                boxes.append(torch.tensor([0., 0., float(W), float(H)], device=mask.device))
            else:
                y0, y1 = ys.min().float(), ys.max().float() + 1.0
                x0, x1 = xs.min().float(), xs.max().float() + 1.0
                boxes.append(torch.stack([x0, y0, x1, y1], dim=0))
        return torch.stack(boxes, dim=0)

    def soft_threshold_and_box(self, attn_grid: torch.Tensor) -> torch.Tensor:
        B, H, W = attn_grid.shape
        device = attn_grid.device

        q = float(self.vs_percentile.item()) / 100.0
        thr = self._quantile(attn_grid, q=q)
        hard_mask = (attn_grid >= thr).to(attn_grid.dtype)
        if bool(self.vs_dilate_half_patch.item()):
            hard_mask = self._dilate_mask(hard_mask, ksize=3, iters=1)

        min_area = int(self.vs_min_area.item())
        keep = (hard_mask.sum(dim=(1, 2)) >= min_area)
        boxes_hard = self._bbox_from_mask(hard_mask)

        if (~keep).any():
            sub = attn_grid[~keep]
            threshold = torch.sigmoid(self.attention_threshold)
            temp = F.softplus(self.temperature)
            w = torch.sigmoid((sub - threshold) * temp)

            y = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
            x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)

            tot = w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            yc = (w * y).sum(dim=(1, 2)) / tot.view(-1)
            xc = (w * x).sum(dim=(1, 2)) / tot.view(-1)
            yvar = ((y - yc.view(-1, 1, 1)) ** 2 * w).sum(dim=(1, 2)) / tot.view(-1)
            xvar = ((x - xc.view(-1, 1, 1)) ** 2 * w).sum(dim=(1, 2)) / tot.view(-1)

            scale = 1.5
            yhalf = torch.sqrt(yvar.clamp_min(1e-8)) * scale
            xhalf = torch.sqrt(xvar.clamp_min(1e-8)) * scale
            y0 = (yc - yhalf).clamp(0, H - 1)
            y1 = (yc + yhalf).clamp_min(y0 + 1.0).clamp(max=float(H))
            x0 = (xc - xhalf).clamp(0, W - 1)
            x1 = (xc + xhalf).clamp_min(x0 + 1.0).clamp(max=float(W))
            boxes_soft = torch.stack([x0, y0, x1, y1], dim=-1)
            boxes_hard[~keep] = boxes_soft

        max_cov = float(self.vs_max_cov.item())
        max_area = max_cov * H * W
        margin = int(self.vs_margin.item())
        x0, y0, x1, y1 = boxes_hard.unbind(-1)
        bw = (x1 - x0).clamp_min(1.0)
        bh = (y1 - y0).clamp_min(1.0)
        area = bw * bh
        over = area > max_area
        if over.any():
            scale = torch.sqrt(max_area / area[over]).view(-1, 1)
            cx = (x0[over] + x1[over]) * 0.5
            cy = (y0[over] + y1[over]) * 0.5
            half_w = bw[over] * 0.5 * scale.squeeze(1)
            half_h = bh[over] * 0.5 * scale.squeeze(1)
            x0[over] = (cx - half_w).clamp(0, W - 1)
            x1[over] = (cx + half_w).clamp_min(x0[over] + 1.0).clamp(max=float(W))
            y0[over] = (cy - half_h).clamp(0, H - 1)
            y1[over] = (cy + half_h).clamp_min(y0[over] + 1.0).clamp(max=float(H))

        x0 = (x0 - margin).clamp(0, W - 1)
        y0 = (y0 - margin).clamp(0, H - 1)
        x1 = (x1 + margin).clamp_min(x0 + 1.0).clamp(max=float(W))
        y1 = (y1 + margin).clamp_min(y0 + 1.0).clamp(max=float(H))

        boxes_patch = torch.stack([x0, y0, x1, y1], dim=-1)
        return boxes_patch

    def _crops_via_grid_sample(self, pixel_values: torch.Tensor, boxes_224: torch.Tensor, out_size=None):
        B, C, H, W = pixel_values.shape
        Hout = Wout = int(self.image_size.item()) if out_size is None else int(out_size)

        u = torch.linspace(-1, 1, Wout, device=pixel_values.device, dtype=torch.float32)
        v = torch.linspace(-1, 1, Hout, device=pixel_values.device, dtype=torch.float32)
        U, V = torch.meshgrid(u, v, indexing='xy')
        U = U.unsqueeze(0).expand(B, Hout, Wout)
        V = V.unsqueeze(0).expand(B, Hout, Wout)

        x0, y0, x1, y1 = boxes_224.unbind(-1)
        bw = (x1 - x0)
        bh = (y1 - y0)

        # --- LOGIC LETTERBOX ---
        max_dim = torch.max(bw, bh) # Lấy chiều lớn nhất để tạo hình vuông
        cx = x0 + bw * 0.5
        cy = y0 + bh * 0.5

        max_dim = max_dim.unsqueeze(-1).unsqueeze(-1)
        cx = cx.unsqueeze(-1).unsqueeze(-1)
        cy = cy.unsqueeze(-1).unsqueeze(-1)

        x = cx + U * 0.5 * max_dim
        y = cy + V * 0.5 * max_dim
        # -----------------------

        x_norm = (x / (W - 1)) * 2 - 1
        y_norm = (y / (H - 1)) * 2 - 1
        grid = torch.stack([x_norm, y_norm], dim=-1)

        crops = F.grid_sample(
            pixel_values.float(), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        return crops

    def _extract_roi_features(self, pixel_values: torch.Tensor, boxes_224: torch.Tensor):
        B, C, H, W = pixel_values.shape
        cnn_size = int(self.cnn_input_size.item())

        crops_vit_norm = self._crops_via_grid_sample(
            pixel_values,
            boxes_224,
            out_size=cnn_size
        )

        vit_mean = self.vit_mean.view(1, C, 1, 1)
        vit_std = self.vit_std.view(1, C, 1, 1)
        cnn_mean = self.cnn_mean.view(1, C, 1, 1)
        cnn_std = self.cnn_std.view(1, C, 1, 1)

        crops_denorm = crops_vit_norm * vit_std + vit_mean
        crops_cnn_norm = (crops_denorm - cnn_mean) / cnn_std

        feats = self.cnn(pixel_values=crops_cnn_norm.to(self.cnn.dtype)).last_hidden_state

        cnn_activation = feats.mean(dim=1)

        B_out, C_out, Hp, Wp = feats.shape
        crop_tokens = feats.permute(0, 2, 3, 1).reshape(B_out, Hp * Wp, C_out)
        return crop_tokens, cnn_activation

    def _proc_params(self, W0, H0):
        size_cfg = getattr(self.vit_processor, "size", int(self.image_size.item()))
        if isinstance(size_cfg, dict):
            if "shortest_edge" in size_cfg:
                tgt = int(size_cfg["shortest_edge"])
                s = tgt / min(W0, H0)
                Wr, Hr = int(round(W0 * s)), int(round(H0 * s))
                sx, sy = s, s
            elif "width" in size_cfg and "height" in size_cfg:
                Wr, Hr = int(size_cfg["width"]), int(size_cfg["height"])
                sx, sy = Wr / W0, Hr / H0
            else:
                tgt = int(self.image_size.item()); s = tgt / min(W0, H0)
                Wr, Hr = int(round(W0 * s)), int(round(H0 * s)); sx, sy = s, s
        else:
            tgt = int(size_cfg)
            if H0 < W0:
                Wr = int(round(W0 * tgt / H0))
                Hr = tgt
            else:
                Wr = tgt
                Hr = int(round(H0 * tgt / W0))
            sx, sy = Wr / W0, Hr / H0


        crop_cfg = getattr(self.vit_processor, "crop_size",
                           {"height": int(self.image_size.item()), "width": int(self.image_size.item())})
        Cw = int(crop_cfg.get("width", int(self.image_size.item())))
        Ch = int(crop_cfg.get("height", int(self.image_size.item())))

        do_center_crop = getattr(self.vit_processor, "do_center_crop", True)
        if do_center_crop:
            left = max(0, (Wr - Cw) // 2); top = max(0, (Hr - Ch) // 2)
        else:
            left, top = 0, 0; Cw, Ch = Wr, Hr

        return dict(W0=W0, H0=H0, Wr=Wr, Hr=Hr, sx=sx, sy=sy,
                    Cw=Cw, Ch=Ch, left=left, top=top)

    def _map_boxes_to_original(self, boxes_224: torch.Tensor, pil_images: Optional[List[Image.Image]]):
        if pil_images is None:
            return boxes_224.detach().cpu().numpy().tolist()
        boxes_list = []
        for b, pil_img in enumerate(pil_images):
            W0, H0 = pil_img.size
            p = self._proc_params(W0, H0)
            box = boxes_224[b].detach().cpu().numpy()

            x0_r = box[0] + p["left"]; y0_r = box[1] + p["top"]
            x1_r = box[2] + p["left"]; y1_r = box[3] + p["top"]

            x0_o = x0_r / p["sx"]; y0_o = y0_r / p["sy"]
            x1_o = x1_r / p["sx"]; y1_o = y1_r / p["sy"]

            x0_o = max(0, min(W0, int(round(x0_o)))); y0_o = max(0, min(H0, int(round(y0_o))))
            x1_o = max(0, min(W0, int(round(x1_o)))); y1_o = max(0, min(H0, int(round(y1_o))))
            boxes_list.append((x0_o, y0_o, x1_o, y1_o))
        return boxes_list

    def forward(
        self,
        img_tokens: torch.Tensor,
        patch_scores: torch.Tensor,
        pixel_values: torch.Tensor,
        return_debug: bool = False,
        pil_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, torch.Tensor]:

        device = pixel_values.device

        attn_summary = img_tokens.mean(dim=1)

        B, L = patch_scores.shape
        g = int(round(math.sqrt(L)))
        if g * g != L:
            raise RuntimeError(f"VisualSearch: patch_scores length {L} is not a perfect square")

        attn_grids = patch_scores.reshape(B, g, g).to(torch.float32)

        flat = attn_grids.reshape(B, -1)
        att_min = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        att_max = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        attn_grids = (attn_grids - att_min) / (att_max - att_min + 1e-8)

        B, g_out, _ = attn_grids.shape
        if g_out != g:
            g = g_out

        boxes_patch = self.soft_threshold_and_box(attn_grids)
        patch_px = float(self.image_size.item()) / float(g)
        boxes_224 = boxes_patch * patch_px

        x0, y0, x1, y1 = boxes_224.unbind(-1)
        x0 = x0.clamp(0, float(self.image_size.item()))
        y0 = y0.clamp(0, float(self.image_size.item()))
        x1 = x1.clamp_min(x0 + 1.0).clamp(max=float(self.image_size.item()))
        y1 = y1.clamp_min(y0 + 1.0).clamp(max=float(self.image_size.item()))
        boxes_224 = torch.stack([x0, y0, x1, y1], dim=-1)

        crop_tokens, cnn_activation = self._extract_roi_features(pixel_values, boxes_224)

        B_img_tok = img_tokens.size(0)
        crop_mask = torch.ones(B_img_tok, crop_tokens.size(1), device=device, dtype=torch.long)

        out = {"crop_tokens": crop_tokens, "crop_mask": crop_mask, "attn_summary": attn_summary}

        if return_debug:
            cnn_size = int(self.cnn_input_size.item())
            debug_crops = self._crops_via_grid_sample(pixel_values, boxes_224, out_size=cnn_size)

            out.update({
                "attn_grids": attn_grids,
                "boxes_224": boxes_224,
                "boxes": self._map_boxes_to_original(boxes_224, pil_images),
                "cnn_activation": cnn_activation,
                "debug_crops": debug_crops,
            })
        return out

    def debug_verify_processor_mapping(self, img_pil: Image.Image):
        with torch.no_grad():
            proc = self.vit_processor(images=[img_pil], return_tensors="pt")
            pv = proc["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
            W0, H0 = img_pil.size
            p = self._proc_params(W0, H0)
            arr = np.array(img_pil)
            resized = cv2.resize(arr, (p["Wr"], p["Hr"]), interpolation=cv2.INTER_CUBIC)
            crop = resized[p["top"]:p["top"]+p["Ch"], p["left"]:p["left"]+p["Cw"]]
            return pv, crop, p

    def _proc_params_resize_only(self, W0: int, H0: int) -> Dict[str, float]:
        size_cfg = getattr(self.vit_processor, "size", int(self.image_size.item()))
        if isinstance(size_cfg, dict):
            if "shortest_edge" in size_cfg:
                tgt = int(size_cfg["shortest_edge"])
                s = tgt / min(W0, H0)
                Wr, Hr = int(round(W0 * s)), int(round(H0 * s)); sx = sy = s
            elif "width" in size_cfg and "height" in size_cfg:
                Wr, Hr = int(size_cfg["width"]), int(size_cfg["height"])
                sx, sy = Wr / W0, Hr / H0
            else:
                tgt = int(self.image_size.item())
                s = tgt / min(W0, H0)
                Wr, Hr = int(round(W0 * s)), int(round(H0 * s)); sx = sy = s
        else:
            tgt = int(size_cfg)
            Wr = Hr = tgt
            sx, sy = Wr / W0, Hr / H0
        return dict(W0=W0, H0=H0, Wr=Wr, Hr=Hr, sx=sx, sy=sy, left=0, top=0, Cw=Wr, Ch=Hr)

    def _heat_on_original(self, img_pil: Image.Image, attn_grid_b: torch.Tensor):
        W0, H0 = img_pil.size
        p = self._proc_params(W0, H0)
        g = attn_grid_b.detach().cpu().numpy()
        heat224 = cv2.resize(g, (p["Cw"], p["Ch"]), interpolation=cv2.INTER_CUBIC)
        heat_rgb = (cm.jet(heat224)[..., :3] * 255).astype(np.uint8)
        canvas = np.zeros((p["Hr"], p["Wr"], 3), dtype=np.uint8)
        y0, y1 = p["top"], p["top"] + p["Ch"]
        x0, x1 = p["left"], p["left"] + p["Cw"]
        canvas[y0:y1, x0:x1] = heat_rgb
        canvas_orig = cv2.resize(canvas, (W0, H0), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(canvas_orig)

    def visualize(self, img_pil: Image.Image, attn_grid_b: torch.Tensor, box_orig):
        import matplotlib.pyplot as plt
        heat_on_orig = self._heat_on_original(img_pil, attn_grid_b)
        overlay = Image.blend(img_pil, heat_on_orig, alpha=0.45)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box_orig, outline=(0, 255, 255), width=3)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.imshow(img_pil);   plt.title("Original Image"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(overlay);   plt.title("Attention + BBox"); plt.axis("off")
        plt.tight_layout(); plt.show()

    def show_crops(self, pil_img: Image.Image, boxes, crop_size=128):
        import matplotlib.pyplot as plt
        crops = []
        for (x0, y0, x1, y1) in boxes:
            crop = pil_img.crop((x0, y0, x1, y1))
            if crop_size: crop = crop.resize((crop_size, crop_size))
            crops.append(crop)
        n = len(crops); cols = min(n, 5); rows = (n + cols - 1) // cols
        plt.figure(figsize=(3*cols, 3*rows))
        for i, c in enumerate(crops):
            plt.subplot(rows, cols, i+1); plt.imshow(c); plt.axis("off")
        plt.suptitle("Cropped Regions"); plt.tight_layout(); plt.show()

    def visualize_on_original(self, pil_img: Image.Image, boxes, color=(255, 0, 0)):
        import matplotlib.pyplot as plt
        overlay = pil_img.copy()
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            draw.rectangle(box, outline=color, width=3)
        plt.figure(figsize=(6, 6)); plt.imshow(overlay); plt.axis("off"); plt.title("Bounding Boxes on Original Image"); plt.show()