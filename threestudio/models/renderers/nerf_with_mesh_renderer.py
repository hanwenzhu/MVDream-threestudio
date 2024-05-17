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
from threestudio.models.renderers.nerf_volume_renderer import NeRFVolumeRenderer
from threestudio.models.renderers.nvdiff_rasterizer import NVDiffRasterizer
from threestudio.utils.misc import get_device
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nerf-with-mesh-renderer")
class NeRFWithMeshRenderer(NeRFVolumeRenderer):
    """Combines an NeRF volume renderer with rendering an explicit Mesh."""

    @dataclass
    class Config(NeRFVolumeRenderer.Config, NVDiffRasterizer.Config):
        mesh_path: str = ""
        # See Mesh.from_path for options
        mesh: dict = field(default_factory=dict)

        geometry_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        geometry_scale: float = 1.0
        geometry_rotation_deg: float = 0.0
        geometry_mask: bool = True

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

    def focus_to_geometry(
        self, points: Float[Tensor, "*N Di"]
    ) -> Float[Tensor, "*N Di"]:
        # Transform points for rendering composed scene to focusing on individual object
        transformed = points
        transformed -= torch.as_tensor(self.cfg.geometry_center).to(transformed)
        transformed /= self.cfg.geometry_scale
        rotation = self.cfg.geometry_rotation_deg * math.pi / 180.0
        transformed = transformed @ torch.as_tensor([
            [math.cos(rotation), math.sin(rotation), 0.0],
            [-math.sin(rotation), math.cos(rotation), 0.0],
            [0.0, 0.0, 1.0]
        ]).to(transformed)
        return transformed

    def geometry_forward(
        self, points: Float[Tensor, "*N Di"], density_only: bool = False, **kwargs
    ) -> Dict[str, Any]:
        focused = self.focus_to_geometry(points)
        if density_only:
            geo_out = {"density": self.geometry.forward_density(focused, **kwargs)}
        else:
            geo_out = self.geometry(focused, **kwargs)
        if self.cfg.geometry_mask:
            # Make density 0 if > radius (to prevent density on unrendered points if points are transformed)
            geo_out["density"][torch.sqrt((focused ** 2).sum(dim=-1))[..., None] > self.geometry.cfg.radius] = 0.0
        return geo_out

    def geometry_forward_density(
        self, points: Float[Tensor, "*N Di"]
    ) -> Float[Tensor, "*N 1"]:
        return self.geometry_forward(points)["density"]

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        bg_color: Optional[Tensor] = None,
        render_mesh: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        assert rays_o.shape == (batch_size, height, width, 3)

        # Step 1: Sample implicit volume and background
        # From nerf_volume_renderer.py:
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            if self.training:
                sigma = self.geometry_forward_density(positions)[..., 0]
            else:
                sigma = chunk_batch(
                    self.geometry_forward_density,
                    self.cfg.eval_chunk_size,
                    positions,
                )[..., 0]
            return sigma

        if not self.cfg.grid_prune:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
        else:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                )

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            geo_out = self.geometry_forward(
                positions, output_normal=self.material.requires_normal
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry_forward,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        if bg_color is None:
            bg_color = comp_rgb_bg.view(batch_size, height, width, -1)
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        weights: Float[Tensor, "Nr 1"]
        weights_, _, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]


        # Step 2: Render mesh and add to background
        if render_mesh:
            # From nvdiff_rasterizer.py:
            v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                self.mesh.v_pos, mvp_mtx
            )
            rast, _ = self.ctx.rasterize(v_pos_clip, self.mesh.t_pos_idx, (height, width))
            mask = rast[..., 3:] > 0

            gb_pos, _ = self.ctx.interpolate_one(self.mesh.v_pos, rast, self.mesh.t_pos_idx)
            
            # We could generate color from self.geometry and self.material (as in nvdiff_rasterizer.py)
            # Instead we generate color directly from mesh information v_rgb
            gb_rgb_fg, _ = self.ctx.interpolate_one(self.mesh.v_rgb, rast, self.mesh.t_pos_idx)

            # Add mesh rendering RGB to background and then the implicit volume RGB
            gb_rgb = torch.lerp(bg_color, gb_rgb_fg, mask.float())
            gb_rgb_fg_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, self.mesh.t_pos_idx)
            gb_rgb_fg_aa = gb_rgb_fg_aa.reshape(batch_size * height * width, -1)

            # Step 3: Zero the weights where the implicit volume is occluded by the mesh
            gb_distances = torch.linalg.norm(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            ).reshape(-1, 1)  # (B*H*W, 1) distance from camera to mesh
            ray_camera_positions = camera_positions[:, None, None, :].expand(
                -1, height, width, -1
            ).reshape(-1, 3)  # (B*H*W, 3) position of camera for each ray
            distances = torch.linalg.norm(
                positions - ray_camera_positions[ray_indices], dim=-1, keepdim=True
            )  # (N, 1) distance from camera to sample position
            # Set occluded weight to 0
            weights[
                mask.reshape(-1, 1)[ray_indices] &  # boolean mask for mesh
                (distances >= gb_distances[ray_indices])  # mesh is closer than the sampling point
            ] = 0.0
        else:
            gb_rgb_fg_aa = bg_color.reshape(batch_size * height * width, -1)
            # We can remove points inside mesh by setting
            #   weights[self.mesh.contains_points(positions)[..., None]] = 0.0
        
        intersection = weights[self.mesh.contains_points(positions)[..., None]]

        # Step 4: Render implicit volume
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        # Step 5: Add the rendered implicit volume to the rendered mesh
        comp_rgb = comp_rgb_fg + gb_rgb_fg_aa * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
            "intersection": intersection,
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry_forward(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.grid_prune:

            def occ_eval_fn(x):
                density = self.geometry_forward_density(x)
                # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                return density * self.render_step_size

            if self.training and not on_load_weights:
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )
