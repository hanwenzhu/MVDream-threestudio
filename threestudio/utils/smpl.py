# import cv2
# import numpy as np
# import pickle
# import smplpytorch.native.webuser.serialization

# # Hot fix: remove dependency on chumpy
# # From smplpytorch.native.webuser.posemapper:
# def lrotmin(p: np.ndarray):
#     p = p.ravel()[3:]
#     return np.concatenate([(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()


# def posemap(s):
#     if s == "lrotmin":
#         return lrotmin
#     else:
#         raise Exception("Unknown posemapping: %s" % (str(s),))


# # Replace serialization.ready_arguments
# def ready_arguments(fname_or_dict):
#     if not isinstance(fname_or_dict, dict):
#         # dd = pickle.load(open(fname_or_dict, "rb"), encoding="latin1")
#         # Here we expect the SMPL model to be fixed by clean_ch.py
#         dd = pickle.load(open(fname_or_dict, "rb"))
#     else:
#         dd = fname_or_dict

#     want_shapemodel = "shapedirs" in dd
#     nposeparms = dd["kintree_table"].shape[1] * 3

#     if "trans" not in dd:
#         dd["trans"] = np.zeros(3)
#     if "pose" not in dd:
#         dd["pose"] = np.zeros(nposeparms)
#     if "shapedirs" in dd and "betas" not in dd:
#         dd["betas"] = np.zeros(dd["shapedirs"].shape[-1])

#     if want_shapemodel:
#         dd["v_shaped"] = dd["shapedirs"].dot(dd["betas"]) + dd["v_template"]
#         v_shaped = dd["v_shaped"]
#         J_tmpx = dd["J_regressor"] @ v_shaped[:, 0]
#         J_tmpy = dd["J_regressor"] @ v_shaped[:, 1]
#         J_tmpz = dd["J_regressor"] @ v_shaped[:, 2]
#         dd["J"] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
#         dd["v_posed"] = v_shaped + dd["posedirs"].dot(posemap(dd["bs_type"])(dd["pose"]))
#     else:
#         dd["v_posed"] = dd["v_template"] + dd["posedirs"].dot(posemap(dd["bs_type"])(dd["pose"]))

#     return dd

# smplpytorch.native.webuser.serialization.ready_arguments = ready_arguments


# # Import after this fix
# from smplpytorch.pytorch.smpl_layer import SMPL_Layer


# Adapted from https://github.com/CalciferZh/SMPL
# (MIT License, Copyright (c) 2018 CalciferZh)
import numpy as np
import pickle
import torch
from torch.nn import Module


class SMPLModel(Module):
    def __init__(self, model_path: str):
        super(SMPLModel, self).__init__()

        with open(model_path, "rb") as f:
            params = pickle.load(f)

        self.register_buffer(
            "J_regressor",
            torch.from_numpy(np.array(params["J_regressor"].todense())).float()
        )
        for name in ("weights", "posedirs", "v_template", "shapedirs"):
            self.register_buffer(
                name,
                torch.from_numpy(params[name]).float()
            )
        self.kintree_table = params["kintree_table"]
        self.register_buffer(
            "faces",
            torch.from_numpy(params["f"]).long()
        )

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues" rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        #r = r.to(self.device)
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)    # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim).to(r)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = torch.eye(3).to(r).unsqueeze(dim=0) + torch.zeros((theta_dim, 3, 3)).to(r)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(x)
        ret = torch.cat((x, ones), dim=0)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros((x.shape[0], 4, 3)).to(x)
        ret = torch.cat((zeros43, x), dim=2)
        return ret

    def forward(self, betas, pose, scale, trans, simplify=False):
        """
                    Construct a compute graph that takes in parameters and outputs a tensor as
                    model vertices. Face indices are also returned as a numpy ndarray.

                    Prameters:
                    ---------
                    pose: Also known as "theta", a [24,3] tensor indicating child joint rotation
                    relative to parent joint. For root joint it"s global orientation.
                    Represented in a axis-angle format.

                    betas: Parameter for model shape. A tensor of shape [10] as coefficients of
                    PCA components. Only 10 components were released by SMPL author.

                    scale: Scaling parameter of shape [].

                    trans: Global translation tensor of shape [3].

                    Return:
                    ------
                    A tensor for vertices, and a numpy ndarray as face indices.

        """
        id_to_col = {self.kintree_table[1, i]: i
                                 for i in range(self.kintree_table.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = (torch.eye(3).to(R_cube).unsqueeze(dim=0) +
                torch.zeros((R_cube.shape[0], 3, 3)).to(R_cube))
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

        results = []
        results.append(
            self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
        )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (R_cube_big[i], torch.reshape(J[i, :] - J[parent[i], :], (3, 1))),
                            dim=1
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=0)
        results = stacked - \
            self.pack(
                torch.matmul(
                    stacked,
                    torch.reshape(
                        torch.cat((J, torch.zeros((24, 1)).to(J)), dim=1),
                        (24, 4, 1)
                    )
                )
            )
        T = torch.tensordot(self.weights, results, dims=([1], [0]))
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((v_posed.shape[0], 1)).to(v_posed)), dim=1
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        v *= scale
        result = v + torch.reshape(trans, (1, 3))
        return result
