"""Anime upscaler based on Anime4K-PyTorch."""

__all__ = ["Upscaler"]

import torch
import torch.nn.functional

from .vendor.anime4k import anime4k

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
                   "low": fast, with acceptable quality
                   "high": slow, with good quality
        """
        if preset not in ("A", "B", "C"):
            raise ValueError(f"Unknown preset '{preset}'; valid: 'A', 'B', 'C'.")
        if quality not in ("low", "high"):
            raise ValueError(f"Unknown quality '{quality}'; valid: 'low', 'high'.")

        self.device = device
        self.dtype = dtype
        self.upscaled_width = upscaled_width
        self.upscaled_height = upscaled_height
        self.preset = preset
        self.quality = quality

        self.last_report_time = None

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

    def upscale(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Upscale `image_tensor` to size target x target, where `target = self.target_size`.

        The RGB data is upscaled using anime4k, and the alpha channel is upscaled bilinearly.

        `image_tensor`: [c, h, w], where c = 3 (RGB) or c = 4 (RGBA).
        """
        c, h, w = image_tensor.shape
        if c == 4:
            rgb_image_tensor = image_tensor[:3, :, :]
        elif c == 3:
            rgb_image_tensor = image_tensor
        else:
            raise ValueError(f"Upscaler.upscale: Expected [c, h, w] image tensor with 3 (RGB) or 4 (RGBA) channels, got {c} instead (shape {image_tensor.shape}).")
        data = rgb_image_tensor.unsqueeze(0).to(self.device).to(self.dtype)
        upscaled_rgb_tensor = self.pipeline(data)[0]

        if c == 3:
            result = upscaled_rgb_tensor
        else:  # c == 4:  # anime4k supports RGB only; upscale the alpha channel manually.
            upscaled_alpha = torch.nn.functional.interpolate(image_tensor[3, :, :].unsqueeze(0).unsqueeze(0),  # [w, h] -> [batch, c, w, h]
                                                             (self.upscaled_height, self.upscaled_width),
                                                             mode="bilinear")
            upscaled_rgba_tensor = torch.cat([upscaled_rgb_tensor, upscaled_alpha[0]], dim=0)  # RGB [3, w, h], alpha [1, w, h]

            result = upscaled_rgba_tensor

        return result
