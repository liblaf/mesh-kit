import trimesh
from mkit import cli
from trimesh import creation


def main() -> None:
    face: trimesh.Trimesh = creation.icosphere(subdivisions=1, radius=1.0)
    face.export("pre-face.ply")
    skull: trimesh.Trimesh = creation.icosphere(subdivisions=1, radius=0.5)
    skull.export("pre-skull.ply")
    point_mask = skull.vertices[:, 1] < 0
    skull.vertices[point_mask] += [0.1, 0.0, 0.0]
    skull.export("post-skull.ply")


if __name__ == "__main__":
    cli.run(main)
