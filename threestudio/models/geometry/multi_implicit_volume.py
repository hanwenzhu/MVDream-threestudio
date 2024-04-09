from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("multi-implicit-volume")
class MultiImplicitVolume(BaseGeometry):

    def configure(
        self,
        geometries: nn.ModuleList
    ) -> None:
        super().configure()
        self.geometries = geometries
        if len(geometries) > 2:
            threestudio.warn("MultiImplicitVolume geometries list longer than 2; not compatible with intersection logic (yet)")

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        if output_normal:
            raise NotImplementedError

        geo_outs = [
            geometry(points, output_normal=output_normal, add_center=False) for geometry in self.geometries
        ]
        # (#self.geometries, *N, 1)
        densities = torch.stack([geo_out["density"] for geo_out in geo_outs], dim=0)

        output = {
            "density": densities.sum(dim=0),
        }

        # weights in (0, 1), equals sigmoid(unactivated densities)
        # (reflects the way weights are calculated: nerfacc.render_weight_from_density)
        weights = 1 - torch.exp(-densities)  # (#self.geometries, *N, 1)

        if "features" in geo_outs[0]:
            # Output weighted sum of features of each component geometry
            # (#self.geometries, *N, Nf)
            features = torch.stack([geo_out["features"] for geo_out in geo_outs], dim=0)
            output.update({"features": (features * weights).sum(dim=0) / weights.sum(dim=0)})

        # TODO this only works for two objects
        output["intersection"] = weights.prod(dim=0)

        return output

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        densities = [geometry.forward_density(points, add_center=False) for geometry in self.geometries]
        return torch.stack(densities, dim=0).sum(dim=0)

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        density = self.forward_density(points)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        **kwargs,
    ) -> "MultiImplicitVolume":
        raise NotImplementedError
