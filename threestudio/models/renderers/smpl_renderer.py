from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("smpl-renderer")
class SMPLRenderer(NVDiffRasterizer):

    # Currently it is basically the same as NVDiffRasterizer except skipping
    # calculating normals when not necessary
    # TODO merge this into nvdiffrasterizer; no need for this separate class

    @dataclass
    class Config(NVDiffRasterizer.Config):
        pass

    cfg: Config

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        if render_rgb:
            # selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            # gb_light_positions = light_positions[:, None, None, :].expand(
            #     -1, height, width, -1
            # )

            # positions = gb_pos[selector]
            # geo_out = self.geometry(positions, output_normal=False)

            # extra_geo_info = {}
            # if self.material.requires_normal:
            #     gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            #     gb_normal = F.normalize(gb_normal, dim=-1)
            #     gb_normal_aa = torch.lerp(
            #         torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            #     )
            #     gb_normal_aa = self.ctx.antialias(
            #         gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            #     )
            #     out.update({"comp_normal": gb_normal_aa})  # in [0, 1]
            #     extra_geo_info["shading_normal"] = gb_normal[selector]
            # if self.material.requires_tangent:
            #     gb_tangent, _ = self.ctx.interpolate_one(
            #         mesh.v_tng, rast, mesh.t_pos_idx
            #     )
            #     gb_tangent = F.normalize(gb_tangent, dim=-1)
            #     extra_geo_info["tangent"] = gb_tangent[selector]

            # rgb_fg = self.material(
            #     viewdirs=gb_viewdirs[selector],
            #     positions=positions,
            #     light_positions=gb_light_positions[selector],
            #     **extra_geo_info,
            #     **geo_out
            # )
            # gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            # gb_rgb_fg[selector] = rgb_fg

            if mesh.v_rgb is None:
                mesh.set_vertex_color(torch.zeros_like(mesh.v_pos))
            gb_rgb_fg, _ = self.ctx.interpolate_one(mesh.v_rgb, rast, mesh.t_pos_idx)
            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

        return out
