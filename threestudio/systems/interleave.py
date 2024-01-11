import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.mvdream import MVDreamSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("interleave-system")
class InterleaveSystem(MVDreamSystem):
    """Interleaves IF guidance and prompt processor systems to MVDream"""

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

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return 

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return 

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        # original loss (sd-xl)
        loss = super().training_step(batch, batch_idx)["loss"]
        
        # deepfloyd IF loss
        out = self(batch)
        prompt_utils = self.deep_floyd_prompt_processor()
        guidance_out = self.deep_floyd_guidance(
            out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
        )

        for name, value in guidance_out.items():
            self.log(f"train/deep_floyd_{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.deep_floyd_loss_cfg[name.replace("loss_", "lambda_")])

        # only consider loss as in magic3d coarse
        # if not self.cfg.refinement:
        if self.C(self.deep_floyd_loss_cfg["lambda_orient"]) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/deep_floyd_loss_orient", loss_orient)
            loss += loss_orient * self.C(self.deep_floyd_loss_cfg["lambda_orient"])

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/deep_floyd_loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.deep_floyd_loss_cfg["lambda_sparsity"])

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/deep_floyd_loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.deep_floyd_loss_cfg["lambda_opaque"])
        # else:
        #     loss_normal_consistency = out["mesh"].normal_consistency()
        #     self.log("train/loss_normal_consistency", loss_normal_consistency)
        #     loss += loss_normal_consistency * self.C(
        #         self.cfg.loss.lambda_normal_consistency
        #     )

        for name, value in self.deep_floyd_loss_cfg.items():
            self.log(f"train_params/deep_floyd_loss_cfg_{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
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
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
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
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
