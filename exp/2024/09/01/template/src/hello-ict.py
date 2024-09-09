import pyvista as pv
from icecream import ic

import mkit


class CLIConfig(mkit.cli.CLIBaseConfig):
    pass


def main(cfg: CLIConfig) -> None:
    ict = mkit.ext.ICTFaceKit()
    ict.narrow_face.save("data/face.ply")
    ict.extract("Eye socket left").save("data/eye-socket.ply")
    ict.extract("Eyeball left").save("data/eyeball.ply")
    ict.extract("Eye blend left").save("data/eye-blend.ply")
    ict.extract("Eye occlusion left").save("data/eye-occlusion.ply")
    face: pv.PolyData = pv.merge(
        [
            ict.narrow_face,
            ict.extract("Eye occlusion left"),
            ict.extract("Eye occlusion right"),
        ]
    )
    ic(face)
    face.save("data/face-filled.ply")


if __name__ == "__main__":
    mkit.cli.run(main)
