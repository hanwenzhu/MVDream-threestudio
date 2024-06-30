from dataclasses import dataclass, field
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *
from threestudio.utils.smpl import SMPLModel


@threestudio.register("smpl")
class SMPL(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        use_feature_network: bool = False
        vertex_color_file: Optional[str] = None
        fix_vertex_color: bool = False
        fix_location: bool = False

        smpl_init_from: Optional[str] = None
        smpl_model_path: str = ""

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.translation: Float[Tensor, "3"]
        self.scale: Float[Tensor, ""]
        # TODO, figure out rotation/orientation?
        self.shape: Float[Tensor, "10"]
        self.pose: Float[Tensor, "72"]

        location_register = (
            self.register_buffer if self.cfg.fix_location else self.register_parameter
        )
        location_register(
            "translation",
            nn.Parameter(torch.zeros(3, dtype=torch.float32))
        )
        location_register(
            "scale",
            nn.Parameter(torch.zeros((), dtype=torch.float32)),
        )

        self.register_parameter(
            "shape",
            nn.Parameter(torch.zeros(10, dtype=torch.float32)),
        )
        self.register_parameter(
            "pose",
            nn.Parameter(torch.zeros(72, dtype=torch.float32)),
        )

        if self.cfg.smpl_init_from is not None:
            if self.cfg.smpl_init_from.endswith(".pkl"):
                with open(self.cfg.smpl_init_from, "rb") as init_file:
                    init_data = pickle.load(init_file)
                    self.translation.data = torch.as_tensor(init_data["global_body_translation"])
                    self.scale.data = torch.as_tensor(init_data["body_scale"][0])
                    # "global_orient" (?)
                    self.shape.data = torch.as_tensor(init_data["betas"][0])
                    self.pose.data = torch.as_tensor(init_data["body_pose"][0])
            else:
                raise NotImplementedError
        
        self.smpl_model = SMPLModel(model_path=self.cfg.smpl_model_path)

        if self.cfg.use_feature_network:
            self.encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        self.vertex_color: Optional[Float[Tensor, "Nv 3"]]
        if self.cfg.vertex_color_file is not None:
            color_register = (
                self.register_buffer if self.cfg.fix_vertex_color else self.register_parameter
            )
            color_register(
                "vertex_color",
                nn.Parameter(
                    torch.from_numpy(np.load(self.cfg.vertex_color_file).astype(np.float32)) / 255.
                )
            )
        else:
            self.vertex_color = None

    def isosurface(self) -> Mesh:
        # This returns the mesh from SMPL weights (not actually isosurface)
        vertices: Float[Tensor, "Nv 3"] = self.smpl_model(
            pose=self.pose.reshape(24, 3),
            betas=self.shape,
            scale=self.scale,
            trans=self.translation,
        )
        mesh = Mesh(vertices, self.smpl_model.faces)
        if self.cfg.use_feature_network:
            points = contract_to_unisphere(vertices, self.bbox)
            enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            mesh.set_vertex_color(features)
        elif self.vertex_color is not None:
            mesh.set_vertex_color(self.vertex_color)
        return mesh

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        if not self.cfg.use_feature_network:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        return {"features": features}

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "SMPL":
        if isinstance(other, SMPL):
            instance = SMPL(cfg, **kwargs)
            instance.translation.data = other.translation.data.clone()
            instance.scale.data = other.scale.data.clone()
            instance.shape.data = other.shape.data.clone()
            instance.pose.data = other.pose.data.clone()

            if not instance.cfg.fix_vertex_color:
                instance.vertex_color.data = other.vertex_color.data.clone()
        # else:
        #     raise TypeError(
        #         f"Cannot create {SMPL.__name__} from {other.__class__.__name__}"
        #     )

        if instance.cfg.use_feature_network and copy_net:
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.feature_network.load_state_dict(
                other.feature_network.state_dict()
            )
        return instance

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not self.cfg.use_feature_network or self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out
