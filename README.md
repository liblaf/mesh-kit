# mesh-kit

## Code Style

### Abbreviations

- disp: displacement
- tet / tetra: tetrahedron

### Denotations

- `E`: Young's modulus
- `f`: force
- `K`: stiffness matrix / rigidity matrix
- `lambda`: Lamé's first parameter
- `mu`: Lamé's second parameter / shear modulus
- `nu`: Poisson's ratio
- `V`: volume

### Naming Convention

Different types of mesh are distinguished by suffixes:

- `mesh_io`: `meshio.Mesh`
- `mesh_pv`: `pyvista.PolyData`
- `mesh_t3`: `pytorch3d.structures.Meshes`
- `mesh_te`: `meshpy.tet.MeshInfo`
- `mesh_ti`: `taichi.MeshInstance`
- `mesh_tr`: `trimesh.Trimesh`

Different types of array are distinguished by suffixes:

- `arr_ti`: `taichi.ScalarField` / `taichi.MatrixField`
- `arr_np`: `numpy.ndarray`
- `arr_ts`: `torch.Tensor`
