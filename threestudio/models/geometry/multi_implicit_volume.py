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
    @dataclass
    class Config(BaseGeometry.Config):
        blob_centers: List[List[float]] = field(default_factory=lambda: [])
        blob_stds: List[float] = field(default_factory=lambda: [])
        blob_invert_x: List[bool] = field(default_factory=lambda: [])
        blob_mask: bool = True
    
    cfg: Config

    def configure(
        self,
        geometries: nn.ModuleList
    ) -> None:
        super().configure()
        for geometry in geometries:
            if not isinstance(geometry, ImplicitVolume):
                raise TypeError("geometries must be ImplicitVolume")
        self.geometries = geometries
        if len(geometries) > 2:
            threestudio.warn("MultiImplicitVolume geometries list longer than 2; not compatible with intersection logic (yet)")

    def focus_points(
        self, points: Float[Tensor, "*N Di"], i
    ) -> Float[Tensor, "*N Di"]:
        # Transform points for rendering composed scene to focusing on individual object
        transformed = points
        transformed -= torch.as_tensor(self.cfg.blob_centers)[i, :].to(transformed)
        transformed /= self.cfg.blob_stds[i] * 2.0
        if self.cfg.blob_invert_x[i]:
            transformed *= torch.as_tensor([-1.0, 1.0, 1.0]).to(transformed)
        return transformed

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False, filter: Optional[int] = None
    ) -> Dict[str, Float[Tensor, "..."]]:
        if output_normal:
            # TODO
            raise NotImplementedError

        geo_outs = self.forward_geometries(points, filter=filter)
        # (#geometries, *N, 1)
        densities = torch.stack([geo_out["density"] for geo_out in geo_outs], dim=0)

        output = {
            "density": densities.sum(dim=0),
        }

        # weights in (0, 1), equals sigmoid(unactivated densities)
        # (reflects the way weights are calculated: nerfacc.render_weight_from_density)
        weights = 1 - torch.exp(-densities)  # (#geometries, *N, 1)

        if "features" in geo_outs[0]:
            # Output weighted sum of features of each component geometry
            # (#geometries, *N, Nf)
            features = torch.stack([geo_out["features"] for geo_out in geo_outs], dim=0)
            output.update({"features": (features * weights).sum(dim=0) / weights.sum(dim=0)})

        # TODO this only works for two objects
        output["intersection"] = weights.prod(dim=0)

        return output

    def forward_geometries(
        self, points: Float[Tensor, "*N Di"], density_only: bool = False, filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        geo_outs = []
        for i, geometry in enumerate(self.geometries):
            if filter is not None and filter != i:
                continue
            focused = self.focus_points(points, i)
            if density_only:
                geo_out = {"density": geometry.forward_density(focused)}
            else:
                geo_out = geometry(focused)
            if self.cfg.blob_mask:
                # Make density 0 if > radius (to prevent density on unrendered points if points are transformed)
                geo_out["density"][torch.sqrt((focused ** 2).sum(dim=-1))[..., None] > geometry.cfg.radius] = 0.0
            geo_outs.append(geo_out)
        return geo_outs

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        geo_outs = self.forward_geometries(points, density_only=True)
        return torch.stack([out["density"] for out in geo_outs], dim=0).sum(dim=0)

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
