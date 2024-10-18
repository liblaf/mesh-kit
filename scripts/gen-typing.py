import argparse
import itertools
from pathlib import Path

from jinja2 import Template


def as_str(x: tuple[str, ...]) -> str:
    return " ".join(x)


DIMS: list[str] = ["2", "3", "4", "5", "6", "E", "M", "N", "T", "V"]
SHAPES: list[str] = [
    "",
    *map(as_str, itertools.product(DIMS, repeat=1)),
    *map(as_str, itertools.product(DIMS, repeat=2)),
]


parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("-o", "--output")
args: argparse.Namespace = parser.parse_args()
fpath: Path = Path(args.file)
template = Template(fpath.read_text())
rendered: str = f"""\
# This file is @generated by {args.file}.
# Do not edit.
"""
rendered += template.render({"shapes": SHAPES})
rendered = rendered.strip() + "\n"
if args.output:
    output: Path = Path(args.output)
    output.write_text(rendered)
else:
    print(rendered, end="")
