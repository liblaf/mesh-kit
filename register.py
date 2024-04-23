import meshio
import numpy as np
import pytorch3d
import pytorch3d.io
import pytorch3d.loss
import pytorch3d.ops
import pytorch3d.structures
import pyvista as pv
import torch
import trimesh
from icecream import ic
from mkit import _log
from mkit import io as _io
from trimesh import bounds

_log.init()
reader = pv.DICOMReader("/home/liblaf/Documents/CT资料/郭小鹤/术前/S30")
data: pv.ImageData = reader.read()
data = data.gaussian_smooth(progress_bar=True)
threshold: float = 300.0
target_pv: pv.PolyData = data.contour([threshold], progress_bar=True)  # pyright: ignore[reportArgumentType]
target_pv.connectivity(extraction_mode="largest", inplace=True, progress_bar=True)
target: trimesh.Trimesh = _io.to_trimesh(target_pv)
target.export("target.ply")

source: trimesh.Trimesh = trimesh.load("pre-skull-gt.ply")
source.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]))
source.export("source.ply")
_, transform = bounds.to_extents(source.bounds)
source.apply_transform(np.linalg.inv(transform))
source.export("source-1.ply")
_, transform = bounds.to_extents(target.bounds)
source.apply_transform(transform)
source.export("source-2.ply")

# matrix, transformed, cost = trimesh.registration.icp(source.vertices, target)
# ic(cost)
# source.apply_transform(matrix)
# source.export("icp.ply")


source_t3 = pytorch3d.structures.Meshes(
    [torch.tensor(source.vertices, dtype=torch.float32)], [torch.tensor(source.faces)]
).cuda()
target_t3 = pytorch3d.structures.Meshes(
    [torch.tensor(target.vertices, dtype=torch.float32)], [torch.tensor(target.faces)]
).cuda()
solution: pytorch3d.ops.points_alignment.ICPSolution = (
    pytorch3d.ops.iterative_closest_point(
        source_t3.verts_padded(),
        target_t3.verts_padded(),
        estimate_scale=True,
        verbose=True,
    )
)
ic(solution.rmse)
source_t3 = pytorch3d.structures.Meshes(solution.Xt, source_t3.faces_padded())
source = trimesh.Trimesh(
    source_t3.verts_list()[0].cpu(), source_t3.faces_list()[0].cpu()
)
source.export("icp.ply")
source_t3 = pytorch3d.structures.Meshes(
    (
        source_t3.verts_padded()
        - torch.tensor(source.center_mass, device="cuda", dtype=torch.float32)
    )
    / source.scale,
    source_t3.faces_padded(),
)
target_t3 = pytorch3d.structures.Meshes(
    (
        target_t3.verts_padded()
        - torch.tensor(source.center_mass, device="cuda", dtype=torch.float32)
    )
    / source.scale,
    target_t3.faces_padded(),
)


deform_verts = torch.zeros(
    source_t3.verts_packed().shape, device="cuda", requires_grad=True
)
optimizer = torch.optim.Adam([deform_verts])
# Number of optimization steps
Niter = 200
# Weight for the chamfer loss
w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 1.0
# Weight for mesh normal consistency
w_normal = 0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.1
# Plot period for the losses
plot_period = 250

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


for i in range(Niter):
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_src_mesh = source_t3.offset_verts(deform_verts)

    # We sample 5k points from the surface of each mesh
    sample_trg = pytorch3d.ops.sample_points_from_meshes(target_t3)
    sample_src = pytorch3d.ops.sample_points_from_meshes(new_src_mesh)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = pytorch3d.loss.chamfer_distance(sample_trg, sample_src)

    # and (b) the edge length of the predicted mesh
    loss_edge = pytorch3d.loss.mesh_edge_loss(new_src_mesh)

    # mesh normal consistency
    loss_normal = pytorch3d.loss.mesh_normal_consistency(new_src_mesh)

    # mesh laplacian smoothing
    loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(
        new_src_mesh, method="uniform"
    )

    # Weighted sum of the losses
    loss = (
        loss_chamfer * w_chamfer
        + loss_edge * w_edge
        + loss_normal * w_normal
        + loss_laplacian * w_laplacian
    )

    # Print the losses
    ic(i, loss.item())

    # Save the losses for plotting
    chamfer_losses.append(w_chamfer * float(loss_chamfer.detach().cpu()))
    edge_losses.append(w_edge * float(loss_edge.detach().cpu()))
    normal_losses.append(w_normal * float(loss_normal.detach().cpu()))
    laplacian_losses.append(w_laplacian * float(loss_laplacian.detach().cpu()))

    # Optimization step
    loss.backward()
    optimizer.step()


source_t3 = source_t3.offset_verts(deform_verts)
meshio.Mesh(
    source_t3.verts_list()[0].detach().cpu().numpy() * source.scale
    + source.center_mass,
    [("triangle", source_t3.faces_list()[0].cpu())],
).write("result.ply")


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(13, 5))
ax = fig.gca()
ax.plot(chamfer_losses, label="chamfer loss")
ax.plot(edge_losses, label="edge loss")
ax.plot(normal_losses, label="normal loss")
ax.plot(laplacian_losses, label="laplacian loss")
ax.legend(fontsize="16")
ax.set_xlabel("Iteration", fontsize="16")
ax.set_ylabel("Loss", fontsize="16")
ax.set_title("Loss vs iterations", fontsize="16")
fig.savefig("loss.pdf")
