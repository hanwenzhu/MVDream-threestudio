import os
from dataclasses import dataclass, field

from omegaconf import OmegaConf
import torch
import torch.nn as nn

import threestudio
from threestudio.systems.mvdream import MVDreamSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("multi-mvdream-with-deepfloyd")
class MultiMVDreamWithDeepFloydSystem(MVDreamSystem):
    """Adds DeepFloyd guidance to MVDream for multiple objects"""
    # Currently, use one DeepFloyd guidance for overall (with self.cfg.prompt)
    # and one MVDream guidance for each sub-object (with self.cfg.prompt_processor and self.cfg.prompts)
    # TODO: refactor and generalize to any number of guidances each specifying which objects to render
    # TODO: generalize to multiple (not just 2) objects

    @dataclass
    class Config(MVDreamSystem.Config):
        prompt: str = ""
        prompts: List[str] = field(default_factory=lambda: [])
    
    cfg: Config

    def configure(self) -> None:
        if self.cfg.geometry_convert_from:
            raise NotImplementedError
        
        if not self.cfg.prompts:
            raise ValueError("Empty system.prompts")

        # self.geometry: a MultiImplicitVolume holding sub-geometries
        if self.cfg.geometry_type != "implicit-volume":
            raise NotImplementedError
        self.geometries = nn.ModuleList([
            threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
            for _ in self.cfg.prompts
        ])
        self.geometry = threestudio.find("multi-implicit-volume")({}, geometries=self.geometries)

        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )

        # self.renderers: renderer for each of geometries
        # self.renderer: renderer for self.geometry
        self.renderers = nn.ModuleList([
            threestudio.find(self.cfg.renderer_type)(
                self.cfg.renderer,
                geometry=geometry,
                material=self.material,
                background=self.background,
            )
            for geometry in self.geometries
        ])
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processors = [
            threestudio.find(self.cfg.prompt_processor_type)(
                {**self.cfg.prompt_processor, "prompt": prompt}
            )
            for prompt in self.cfg.prompts
        ]
        self.prompt_utils = [prompt_processor() for prompt_processor in self.prompt_processors]
    
    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.deep_floyd_prompt_processor = threestudio.find("deep-floyd-prompt-processor")({
            "pretrained_model_name_or_path": "DeepFloyd/IF-I-XL-v1.0",
            "prompt": self.cfg.prompt,
        })
        self.deep_floyd_guidance = threestudio.find("deep-floyd-guidance")({
            "pretrained_model_name_or_path": "DeepFloyd/IF-I-XL-v1.0",
            "weighting_strategy": "uniform",
            "guidance_scale": 20.,
            "min_step_percent": 0.02,
            "max_step_percent": 0.98,
        })
        # e.g. self.cfg.loss in `Magic3d`
        self.deep_floyd_loss_cfg = OmegaConf.create({
            "lambda_sds": 1.,
            "lambda_orient": [0, 10., 1000., 5000],
            "lambda_sparsity": 1.,
            "lambda_opaque": 0.,
        })

    def training_step(self, batch, batch_idx):
        # original loss (sd-xl)
        original_loss = 0.0
        # super().training_step uses self.renderer, self.guidance, self.prompt_utils
        for renderer, prompt_utils in zip(self.renderers, self.prompt_utils):
            original_loss += self.get_loss(batch, renderer, self.guidance, prompt_utils)

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
