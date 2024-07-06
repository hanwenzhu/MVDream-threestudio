from dataclasses import dataclass, field

import math
import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.models.renderers.smpl_renderer import SMPLRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("smpl-with-mesh-renderer")
class SMPLWithMeshRenderer(SMPLRenderer):
    """Combines an SMPL renderer with rendering an explicit Mesh."""

    @dataclass
    class Config(SMPLRenderer.Config, NVDiffRasterizer.Config):
        mesh_path: str = ""
        # See Mesh.from_path for options
        mesh: dict = field(default_factory=dict)

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.mesh = Mesh.from_path(
            self.cfg.mesh_path, get_device(), **self.cfg.mesh
        )
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_mesh: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        # Merge SMPL mesh and self.mesh
        smpl_mesh = self.geometry.isosurface()
        smpl_rgb: Float[Tensor, "B Nvs 3"]
        if smpl_mesh.v_rgb is not None:
            smpl_rgb = smpl_mesh.v_rgb
        else:
            # TODO, the logic in like nvdiff_rasterizer makes more sense
            # But for that we need a mask that selects the SMPL only
            geo_out = self.geometry(smpl_mesh.v_pos, output_normal=False)
            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = smpl_mesh.v_nrm
            if self.material.requires_tangent:
                extra_geo_info["tangent"] = smpl_mesh.v_tng
            viewdirs = F.normalize(
                smpl_mesh.v_pos[None, :, :] - camera_positions[:, None, :], dim=-1
            )
            lightpos = light_positions[:, None, :].expand(
                -1, smpl_mesh.v_pos.shape[0], -1
            )
            smpl_rgb = self.material(
                viewdirs=viewdirs,
                positions=smpl_mesh.v_pos,
                light_positions=lightpos,
                **extra_geo_info,
                **geo_out
            )
            
        mesh: Mesh
        rgb: Float[Tensor, "B Nv 3"]
        if render_mesh:
            v_pos = torch.cat([
                smpl_mesh.v_pos, self.mesh.v_pos
            ], dim=0)
            t_pos_idx = torch.cat([
                smpl_mesh.t_pos_idx,
                # Offset indices by number of vertices of first half
                self.mesh.t_pos_idx + smpl_mesh.v_pos.shape[0]
            ], dim=0)
            mesh = Mesh(v_pos, t_pos_idx)
            if self.mesh.v_rgb is None:
                threestudio.warn("Given mesh has no color, using all black")
                self.mesh.set_vertex_color(torch.zeros_like(self.mesh.v_pos))
            rgb = torch.cat([
                smpl_rgb, self.mesh.v_rgb[None, :, :].expand(batch_size, -1, -1)
            ], dim=1)
        else:
            mesh = smpl_mesh
            rgb = smpl_rgb

        # Render mesh
        # From nvdiff_rasterizer.py:
        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)
        
        # We generate color directly from mesh information v_rgb
        # gb_rgb_fg, _ = self.ctx.interpolate_one(mesh.v_rgb, rast, mesh.t_pos_idx)
        gb_rgb_fg, _ = self.ctx.interpolate(rgb, rast, mesh.t_pos_idx)
        gb_rgb_bg = self.background(dirs=rays_d)

        # Add mesh rendering RGB to background and then the implicit volume RGB
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = self.ctx.antialias(
            gb_rgb, rast, v_pos_clip, mesh.t_pos_idx
        )

        out = {
            "comp_rgb": gb_rgb_aa,
            "comp_rgb_fg": gb_rgb_fg,
            "comp_rgb_bg": gb_rgb_bg,
            "opacity": mask_aa,
            "depth": 0.0,  # TODO
            "z_variance": 0.0,  # TODO
            "intersection": 0.0,  # TODO
            "mesh_occlusion": 0.0,  # TODO
        }

        return out
