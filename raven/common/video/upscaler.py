"""Anime upscaler based on Anime4K-PyTorch.

This module is licensed under the MIT license (same license as Anime4K).
"""

__all__ = ["Upscaler"]

import threading

import torch
import torch.nn.functional

from ...vendor.anime4k import anime4k

class Upscaler:
    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 upscaled_width: int,
                 upscaled_height: int,
                 preset: str = "C",
                 quality: str = "low") -> None:
        """
        `upscaled_width`, `upscaled_height`: Target (output) image size.
        `preset`: One of "A", "B", or "C".
                  These roughly correspond to the presets of Anime4K:
                      https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
        `quality`: One of:
                   "low": fast, with acceptable quality (Anime4K small models)
                   "high": slow, with good quality (Anime4K larger models)
                   "bilinear": very fast, basic bilinear interpolation (no Anime4K)
                   "bicubic": very fast, bicubic interpolation (no Anime4K, slightly sharper than bilinear)
        """
        if preset not in ("A", "B", "C"):
            raise ValueError(f"Unknown preset '{preset}'; valid: 'A', 'B', 'C'.")
        if quality not in ("low", "high", "bilinear", "bicubic"):
            raise ValueError(f"Unknown quality '{quality}'; valid: 'low', 'high', 'bilinear', 'bicubic'.")

        self.device = device
        self.dtype = dtype
        self.upscaled_width = upscaled_width
        self.upscaled_height = upscaled_height
        self.preset = preset
        self.quality = quality

        # Guards both `upscale` and `reconfigure_output_size`. The upscaler is typically called from a
        # single render thread, but the output size may need updating from a settings-handler thread
        # (e.g. when the server's crop + upscale pipeline reorder wants the output to track the
        # cropped-then-upscaled size); the lock serializes those against an in-flight `upscale` call.
        self._lock = threading.Lock()

        if quality in ("bilinear", "bicubic"):
            self.pipeline = None  # bypass Anime4K; use torch.nn.functional.interpolate
            return

        # For the models, see `anime4k` for explanation, and `anime4k.model_dict` for available choices.
        if preset == "A":  # Restore -> Upscale -> Upscale
            self.step1_model = "Restore_VL" if quality == "high" else "Restore_S"
        elif preset == "B":  # Restore_Soft -> Upscale -> Upscale
            self.step1_model = "Restore_Soft_VL" if quality == "high" else "Restore_Soft_S"
        else:  # preset == "C":  # Upscale_Denoise -> Upscale
            self.step1_model = "Upscale_Denoise_VL" if quality == "high" else "Upscale_Denoise_S"
        self.step2_model = "Upscale_M" if quality == "high" else "Upscale_S"

        if preset in ("A", "B"):
            self.pipeline = anime4k.Anime4KPipeline(
                anime4k.ClampHighlight(),
                anime4k.create_model(self.step1_model),
                anime4k.create_model(self.step2_model),
                anime4k.AutoDownscalePre(4),  # auto scale down so that when we upscale once, we hit target res
                anime4k.create_model(self.step2_model),
                screen_width=upscaled_width, screen_height=upscaled_height,
                final_stage_upscale_mode="bilinear"  # bilinear is usually sufficient here
            ).to(self.device).to(self.dtype)
        else:  # preset == "C":
            self.pipeline = anime4k.Anime4KPipeline(
                anime4k.ClampHighlight(),
                anime4k.create_model(self.step1_model),
                anime4k.AutoDownscalePre(4),  # auto scale down so that when we upscale once, we hit target res
                anime4k.create_model(self.step2_model),
                screen_width=upscaled_width, screen_height=upscaled_height,
                final_stage_upscale_mode="bilinear"  # bilinear is usually sufficient here
            ).to(self.device).to(self.dtype)

    def reconfigure_output_size(self, new_width: int, new_height: int) -> None:
        """Change the upscaler's target output dimensions in place, without rebuilding the pipeline.

        The neural-net weights don't depend on output size — the pipeline's `screen_width`/`screen_height`
        attributes are read at forward time, and `AutoDownscalePre` adapts to any input size. So this is a
        cheap update, safe to call per-frame when the expected output size tracks a dynamic input (e.g.
        after a server-side crop step whose bbox changed).

        No-op if the size is unchanged.
        """
        with self._lock:
            if new_width == self.upscaled_width and new_height == self.upscaled_height:
                return
            self.upscaled_width = new_width
            self.upscaled_height = new_height
            if self.pipeline is not None:
                self.pipeline.screen_width = new_width
                self.pipeline.screen_height = new_height

    def upscale(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Upscale `image_tensor` to `(upscaled_height, upscaled_width)`.

        The RGB data is upscaled using Anime4K, and the alpha channel is upscaled bilinearly.

        `image_tensor`: [c, h, w], where c = 3 (RGB) or c = 4 (RGBA).
        """
        c, h, w = image_tensor.shape
        if c not in (3, 4):
            raise ValueError(f"Upscaler.upscale: Expected [c, h, w] image tensor with 3 (RGB) or 4 (RGBA) channels, got {c} instead (shape {image_tensor.shape}).")

        # Hold the lock across the whole forward pass to serialize against `reconfigure_output_size`.
        with self._lock:
            target_size = (self.upscaled_height, self.upscaled_width)

            if self.pipeline is None:
                # Bypass mode: bilinear or bicubic interpolation, no Anime4K.
                # Alpha always uses bilinear — bicubic's negative lobes cause
                # ringing at silhouette edges (Gibbs phenomenon).
                if c == 3 or self.quality == "bilinear":
                    return torch.nn.functional.interpolate(image_tensor.unsqueeze(0),
                                                           target_size,
                                                           mode=self.quality,
                                                           align_corners=False)[0]
                upscaled_rgb = torch.nn.functional.interpolate(image_tensor[:3, :, :].unsqueeze(0),
                                                               target_size,
                                                               mode="bicubic",
                                                               align_corners=False)[0]
                upscaled_alpha = torch.nn.functional.interpolate(image_tensor[3:4, :, :].unsqueeze(0),
                                                                 target_size,
                                                                 mode="bilinear")[0]
                return torch.cat([upscaled_rgb, upscaled_alpha], dim=0)

            # Anime4K pipeline: operates on RGB only
            rgb_image_tensor = image_tensor[:3, :, :] if c == 4 else image_tensor
            data = rgb_image_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            with torch.inference_mode():
                upscaled_rgb_tensor = self.pipeline(data)[0]

            if c == 3:
                return upscaled_rgb_tensor

            # Anime4K supports RGB only; upscale the alpha channel with bilinear interpolation.
            upscaled_alpha = torch.nn.functional.interpolate(image_tensor[3:4, :, :].unsqueeze(0),
                                                             target_size,
                                                             mode="bilinear")[0]
            return torch.cat([upscaled_rgb_tensor, upscaled_alpha], dim=0)
