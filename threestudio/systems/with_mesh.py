import os
from dataclasses import dataclass, field

from omegaconf import OmegaConf
import torch
import torch.nn as nn

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("with-mesh")
class WithMesh(BaseLift3DSystem):
    """Trains an object in relation to a given mesh, with the overall scene supervised by DeepFloyd."""
    @dataclass
    class Config(BaseLift3DSystem.Config):
        composed_prompt_processor_type: str = ""
        composed_prompt_processor: dict = field(default_factory=dict)

        composed_guidance_type: str = ""
        composed_guidance: dict = field(default_factory=dict)

        composed_loss: dict = field(default_factory=dict)
    
    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        if self.cfg.renderer_type != "nerf-with-mesh-renderer":
            threestudio.warn("system.renderer_type is not nerf-with-mesh-renderer")

        # TODO this should be in on_fit_start, if not for mvdream
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.guidance_type == "multiview-diffusion-guidance":
            self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def on_load_checkpoint(self, checkpoint):
        if self.cfg.guidance_type == "multiview-diffusion-guidance":
            for k in list(checkpoint['state_dict'].keys()):
                if k.startswith("guidance."):
                    return
            guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
            checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
            return

    def on_save_checkpoint(self, checkpoint):
        if self.cfg.guidance_type == "multiview-diffusion-guidance":
            for k in list(checkpoint['state_dict'].keys()):
                if k.startswith("guidance."):
                    checkpoint['state_dict'].pop(k)
            return

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.composed_prompt_processor = threestudio.find(
            self.cfg.composed_prompt_processor_type
        )(self.cfg.composed_prompt_processor)
        self.composed_guidance = threestudio.find(self.cfg.composed_guidance_type)(
            self.cfg.composed_guidance
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        # loss of individual object
        loss = 0.0
        out = self({**batch, "render_mesh": False})  # TODO also transform the nerf
        guidance_out = self.guidance(
            out["comp_rgb"], self.prompt_utils, **batch, rgb_as_latents=False
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # loss of composed scene
        composed_loss = 0.0
        out = self(batch)
        prompt_utils = self.composed_prompt_processor()
        guidance_out = self.composed_guidance(
            out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
        )

        for name, value in guidance_out.items():
            self.log(f"train/composed_{name}", value)
            if name.startswith("loss_"):
                composed_loss += value * self.C(self.cfg.composed_loss[name.replace("loss_", "lambda_")])

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/composed_loss_sparsity", loss_sparsity)
        composed_loss += loss_sparsity * self.C(self.cfg.composed_loss["lambda_sparsity"])

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/composed_loss_opaque", loss_opaque)
        composed_loss += loss_opaque * self.C(self.cfg.composed_loss["lambda_opaque"])

        for name, value in self.cfg.composed_loss.items():
            self.log(f"train_params/composed_{name}", self.C(value))

        lambda_if = self.C(self.cfg.loss["lambda_if"])
        loss = (1 - lambda_if) * loss + lambda_if * composed_loss

        return {"loss": loss}

    # TODO
    def validation_step(self, batch, batch_idx):
        def run_validation(name, batch):
            out = self(batch)
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-{name}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name=f"validation_step-{name}",
                step=self.true_global_step,
            )
        run_validation("no_mesh", {**batch, "render_mesh": False})
        run_validation("with_mesh", batch)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        def run_test(name, batch):
            out = self(batch)
            self.save_image_grid(
                f"it{self.true_global_step}-test-{name}/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name=f"test_step-{name}",
                step=self.true_global_step,
            )
        run_test("no_mesh", {**batch, "render_mesh": False})
        run_test("with_mesh", batch)

    def on_test_epoch_end(self):
        for name in ("no_mesh", "with_mesh"):
            self.save_img_sequence(
                f"it{self.true_global_step}-test-{name}",
                f"it{self.true_global_step}-test-{name}",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name=f"test-{name}",
                step=self.true_global_step,
            )
