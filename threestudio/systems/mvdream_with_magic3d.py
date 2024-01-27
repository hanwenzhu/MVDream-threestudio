import os
from dataclasses import dataclass, field

from omegaconf import OmegaConf
import torch

import threestudio
from threestudio.systems.mvdream import MVDreamSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("mvdream-with-magic3d")
class MVDreamWithMagic3DSystem(MVDreamSystem):
    """Sums gradient updates from MVDream and Magic3D"""

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
    
    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.deep_floyd_prompt_processor = threestudio.find("deep-floyd-prompt-processor")({
            "pretrained_model_name_or_path": "DeepFloyd/IF-I-XL-v1.0",
            "prompt": self.cfg.prompt_processor["prompt"],
        })
        self.deep_floyd_guidance = threestudio.find("deep-floyd-guidance")({
            "pretrained_model_name_or_path": "DeepFloyd/IF-I-XL-v1.0",
            "weighting_strategy": "uniform",
            "guidance_scale": 20.,
            "min_step_percent": 0.02,
            "max_step_percent": 0.98,
        })
        # i.e. self.cfg.loss in `Magic3d`
        self.deep_floyd_loss_cfg = OmegaConf.create({
            "lambda_sds": 1.,
            "lambda_orient": [0, 10., 1000., 5000],
            "lambda_sparsity": 1.,
            "lambda_opaque": 0.,
        })

    def training_step(self, batch, batch_idx):
        # original loss (sd-xl)
        original_loss = super().training_step(batch, batch_idx)["loss"]
        
        # deepfloyd IF loss
        deep_floyd_loss = 0.0
        out = self(batch)
        prompt_utils = self.deep_floyd_prompt_processor()
        guidance_out = self.deep_floyd_guidance(
            out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
        )

        for name, value in guidance_out.items():
            self.log(f"train/deep_floyd_{name}", value)
            if name.startswith("loss_"):
                deep_floyd_loss += value * self.C(self.deep_floyd_loss_cfg[name.replace("loss_", "lambda_")])

        # only consider loss as in magic3d coarse
        # if not self.cfg.refinement:
    
        # no normal
        # if self.C(self.deep_floyd_loss_cfg["lambda_orient"]) > 0:
        #     if "normal" not in out:
        #         raise ValueError(
        #             "Normal is required for orientation loss, no normal is found in the output."
        #         )
        #     loss_orient = (
        #         out["weights"].detach()
        #         * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
        #     ).sum() / (out["opacity"] > 0).sum()
        #     self.log("train/deep_floyd_loss_orient", loss_orient)
        #     deep_floyd_loss += loss_orient * self.C(self.deep_floyd_loss_cfg["lambda_orient"])

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/deep_floyd_loss_sparsity", loss_sparsity)
        deep_floyd_loss += loss_sparsity * self.C(self.deep_floyd_loss_cfg["lambda_sparsity"])

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/deep_floyd_loss_opaque", loss_opaque)
        deep_floyd_loss += loss_opaque * self.C(self.deep_floyd_loss_cfg["lambda_opaque"])
        # else:
        #     loss_normal_consistency = out["mesh"].normal_consistency()
        #     self.log("train/loss_normal_consistency", loss_normal_consistency)
        #     loss += loss_normal_consistency * self.C(
        #         self.cfg.loss.lambda_normal_consistency
        #     )

        for name, value in self.deep_floyd_loss_cfg.items():
            self.log(f"train_params/deep_floyd_loss_cfg_{name}", self.C(value))

        lambda_if = self.C(self.cfg.loss["lambda_if"])
        loss = (1 - lambda_if) * original_loss + lambda_if * deep_floyd_loss
        return {"loss": loss}
