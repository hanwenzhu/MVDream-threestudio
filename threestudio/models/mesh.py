from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import threestudio
from threestudio.utils.ops import dot, scale_tensor
from threestudio.utils.typing import *


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], **kwargs
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._t_tex_idx: Optional[Float[Tensor, "Nf 3"]] = None
        self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]) -> Mesh:
        if self.requires_grad:
            threestudio.debug("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        threestudio.debug(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        threestudio.debug(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        threestudio.debug(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            threestudio.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self):
        if self._t_tex_idx is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._t_tex_idx

    @property
    def v_rgb(self):
        return self._v_rgb

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        threestudio.info("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(
            self.v_pos.detach().cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        atlas.generate(co, po)
        vmapping, indices, uvs = atlas.get_mesh(0)
        vmapping = (
            torch.from_numpy(
                vmapping.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        uvs = torch.from_numpy(uvs).to(self.v_pos.device).float()
        indices = (
            torch.from_numpy(
                indices.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        return uvs, indices

    def unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        self._v_tex, self._t_tex_idx = self._unwrap_uv(
            xatlas_chart_options, xatlas_pack_options
        )

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges

    def normal_consistency(self) -> Float[Tensor, ""]:
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        nc = (
            1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        ).mean()
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]
        F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(
            dim=1
        )
        adj_values = torch.ones(adj.shape[1]).to(verts)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def laplacian(self) -> Float[Tensor, ""]:
        with torch.no_grad():
            L = self._laplacian_uniform()
        loss = L.mm(self.v_pos)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss

    def contains_points(self, points: Float[Tensor, "*N 3"]) -> Bool[Tensor, "*N"]:
        if "bounds" not in self.extras or "occupancies" not in self.extras:
            # TODO, convert to trimesh and calculate bounds and occupancies like in from_path
            raise NotImplementedError

        # self.extras["bounds"] contains the bbox of the mesh
        # self.extras["occupancices"] is of shape (X, Y, Z), where occupancies[x, y, z] contains
        # whether a test point at [x / (X - 1), y / (Y - 1), z / (Z - 1)]
        # (then scaled to the bounding box) is in the mesh
        contains = torch.zeros_like(points[..., 0], dtype=torch.bool)
        bounds = torch.as_tensor(self.extras["bounds"]).to(points)
        occupancies = torch.as_tensor(self.extras["occupancies"]).to(contains)

        contracted = scale_tensor(points, bounds, (0.0, 1.0))
        in_bbox = (0.0 <= contracted).all(dim=-1) & (contracted <= 1.0).all(dim=-1)
        occupancies_shape = torch.as_tensor(occupancies.shape).to(contracted)
        # A point is in the mesh if its closest test point is
        # We convert contracted points in the bbox (in [0, 1]^3) to the test point at
        # round([x*(X-1), y*(Y-1), z*(Z-1)])
        contracted_indices = torch.round(contracted[in_bbox] * (occupancies_shape - 1)).long()
        contains[in_bbox] = occupancies[
            contracted_indices[..., 0], contracted_indices[..., 1], contracted_indices[..., 2]
        ]
        return contains

    def sdf(self, points: Float[Tensor, "*N 3"]) -> Float[Tensor, "*N"]:
        """Returns the (differentiable) signed distances to mesh"""
        if "bounds" not in self.extras or "sdf" not in self.extras:
            # TODO
            raise NotImplementedError

        points_shape = points.shape
        bounds = torch.as_tensor(self.extras["bounds"]).to(points)
        sdf = torch.as_tensor(self.extras["sdf"]).to(points)
        
        # scale to [-1., 1.] for grid_sample
        contracted = scale_tensor(points, bounds, (-1.0, 1.0))

        # interpolate value of 
        sdf_values = F.grid_sample(
            sdf[None, None, ...],
            contracted.view(1, -1, 1, 1, 3),
            padding_mode="border"
        )
        return sdf_values.view(points_shape[:-1])

    @classmethod
    def from_path(
        cls,
        file_path: str,
        device: torch.device,
        # if the model is Y-up instead of Z-up
        # (TODO it may be easier to ask the user to specify a 4x4 matrix instead for the following)
        y_up: bool = True,
        normalize: bool = True,
        scale: Optional[List[float]] = None,
        # Rotation about Z-axis (up)
        rotation_deg: Optional[float] = None,
        # Rotation about X-axis
        tilt_deg: Optional[float] = None,
        translation: Optional[List[float]] = None,
        occupancy_resolution: int = 128,
    ) -> Mesh:
        # TODO(thomas): use the format of tetrahedra-sdf-grid.configure
        import trimesh

        mesh = trimesh.load(file_path, force="mesh")

        if y_up:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        if normalize:
            mesh.apply_translation(-mesh.centroid)
            # Can also use mesh.scale for the max diameter
            mesh.apply_scale(1 / np.linalg.norm(mesh.vertices, axis=1).mean())
        if scale is not None:
            mesh.apply_scale(scale)
        if rotation_deg is not None:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(rotation_deg * np.pi / 180.0, [0, 0, 1]))
        if tilt_deg is not None:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(tilt_deg * np.pi / 180.0, [1, 0, 0]))
        if translation is not None:
            mesh.apply_translation(translation)

        obj = cls(
            v_pos=torch.from_numpy(mesh.vertices).float().to(device),
            t_pos_idx=torch.from_numpy(mesh.faces).to(device)
        )

        # Not currently used
        # obj._v_nrm = torch.from_numpy(mesh.vertex_normals).float().to(device)
        # obj._edges = torch.from_numpy(mesh.edges).to(device)

        # Not currently used, but could use
        # if mesh.visual.kind == "texture":
        #     obj._v_tex = torch.from_numpy(mesh.visual.uv).float().to(device)
        #     obj._t_tex_idx = obj.t_pos_idx

        # Color vertices (converting from a texture map)
        if mesh.visual.kind == "texture":
            color = mesh.visual.to_color()
        else:
            color = mesh.visual
        obj.set_vertex_color((
            # Drop alpha channel
            torch.from_numpy(color.vertex_colors[..., :3]).float() / 255.0
        ).to(device))

        # (2, 3) AABB bounding box
        obj.add_extra("bounds", mesh.bounds)

        # We test if each point in a 3D grid is within the mesh and store in occupancies:
        # occupancies[x, y, z]: for p = (x / 127, y / 127, z / 127), which is in [0, 1]^3
        # we scale p to a point in the bounding box and test if p is contained in the mesh
        test_points = np.stack(np.meshgrid(
            np.linspace(mesh.bounds[0, 0], mesh.bounds[1, 0], occupancy_resolution),
            np.linspace(mesh.bounds[0, 1], mesh.bounds[1, 1], occupancy_resolution),
            np.linspace(mesh.bounds[0, 2], mesh.bounds[1, 2], occupancy_resolution),
            indexing="ij"
        ), axis=-1)

        # For occupancy testing we use trimesh contains_points,
        # which is much faster with embree installed (and pyembree or embreex bindings)
        if not trimesh.ray.has_embree:
            threestudio.warn(
                "Embree is not installed for trimesh, so occupancy testing will be slow and may trigger OOM. "
                "You should install Embree and then either pyembree or embreex. "
                "See trimesh docs for more details."
            )
        threestudio.info(
            "Testing occupancy for mesh. If this takes too long, consider lowering triangle count "
            "or decreasing occupancy_resolution"
        )
        occupancies = mesh.ray.contains_points(
            test_points.reshape(-1, 3)
        ).reshape(test_points.shape[:3])
        # (128, 128, 128)
        obj.add_extra("occupancies", occupancies)

        # Test SDF
        from pysdf import SDF
        sdf_fun = SDF(mesh.vertices, mesh.faces)
        sdf = sdf_fun(test_points.reshape(-1, 3)).reshape(test_points.shape[:3])
        # (128, 128, 128)
        obj.add_extra("sdf", sdf)

        return obj
