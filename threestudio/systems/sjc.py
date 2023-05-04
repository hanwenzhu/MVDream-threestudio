from dataclasses import dataclass, field

import numpy as np
import torch

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

__all__ = ["ScoreJacobianChaining"]


@threestudio.register("sjc-system")
class ScoreJacobianChaining(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "volume-grid"
        geometry: dict = field(default_factory=dict)
        material_type: str = "diffuse-with-point-light-material"
        material: dict = field(default_factory=dict)
        background_type: str = "neural-environment-map-background"
        background: dict = field(default_factory=dict)
        renderer_type: str = "nerf-volume-renderer"
        renderer: dict = field(default_factory=dict)
        guidance_type: str = "-diffusion-guidance"
        guidance: dict = field(default_factory=dict)
        prompt_processor_type: str = "dreamfusion-prompt-processor"
        prompt_processor: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
        # self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any], decode: bool = False) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        out = {
            **render_out,
        }
        if decode:
            out["decoded_rgb"] = self.guidance.decode_latents(
                out["comp_rgb"].permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
        return out

    def on_fit_start(self) -> None:
        """
        Initialize guidance and prompt processor in this hook:
        (1) excluded from optimizer parameters (this hook executes after optimizer is initialized)
        (2) only used in training
        To avoid being saved to checkpoints, see on_save_checkpoint below.
        """
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )

    def training_step(self, batch, batch_idx):
        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)
        guidance_out = self.guidance(
            out["comp_rgb"], text_embeddings, rgb_as_latents=True
        )

        loss = 0.0
        loss += guidance_out["sds"] * self.C(self.cfg.loss.lambda_sds)

        loss_emptiness = (
            self.C(self.cfg.loss.lambda_emptiness)
            * torch.log(1 + self.cfg.loss.emptiness_scale * out["weights"]).mean()
        )

        self.log("train/loss_emptiness", loss_emptiness)
        loss += loss_emptiness

        # DONT USE THIS LOSS, I think it is wrong
        if self.C(self.cfg.loss.lambda_depth) > 0:
            _, h, w, _ = out["comp_rgb"].shape
            comp_depth = (out["depth"] + 10 * (1 - out["opacity"])).squeeze(-1)
            center_h = int(self.cfg.loss.center_ratio * h)
            center_w = int(self.cfg.loss.center_ratio * w)
            border_h = (h - center_h) // 2
            border_w = (h - center_w) // 2
            center_depth = comp_depth[
                ..., border_h : border_h + center_h, border_w : border_w + center_w
            ]
            center_depth_mean = center_depth.mean()
            border_depth_mean = (comp_depth.sum() - center_depth.sum()) / (
                h * w - center_h * center_w
            )
            loss_depth = -torch.log(
                center_depth_mean - border_depth_mean + 1e-6
            ) * self.C(self.cfg.loss.lambda_depth)

            self.log("train/loss_depth", loss_depth)
            if center_depth_mean > border_depth_mean:
                loss += loss_depth

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def vis_depth(self, pred_depth):
        depth = pred_depth.detach().cpu().numpy()
        depth = np.log(1.0 + depth + 1e-12) / np.log(1 + 10.0)
        return depth

    def validation_step(self, batch, batch_idx):
        out = self(batch, decode=True)
        comp_depth = out["depth"] + 10 * (1 - out["opacity"])  # 10 for background
        vis_depth = self.vis_depth(comp_depth.squeeze(-1))

        self.save_image_grid(
            f"it{self.global_step}-{batch_idx}.png",
            [
                {
                    "type": "rgb",
                    "img": out["decoded_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
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
            ]
            + [
                {
                    "type": "grayscale",
                    "img": vis_depth[0],
                    "kwargs": {"cmap": "spectral", "data_range": (0, 1)},
                },
            ],
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch, decode=True)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch_idx}.png",
            [
                {
                    "type": "rgb",
                    "img": out["decoded_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
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
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove stable diffusion weights
        # TODO: better way?
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k.split(".")[0] not in ["prompt_processor", "guidance"]
        }
        return super().on_save_checkpoint(checkpoint)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # debug use
        pass
        # from lightning.pytorch.utilities import grad_norm
        # norms = grad_norm(self.geometry, norm_type=2)
        # print(norms)