import pathlib

import pytest
from PIL import Image

from .conftest import python


@pytest.fixture(scope="module")
def analysis_dir(temp_dir):
    return temp_dir / "analysis"


def test_render_prototypes(analysis_dir, shapes_2d_squeezenet1_1_path):
    stdout, _ = python(
        f"-u -m protopnet viz render-prototypes --model-path={shapes_2d_squeezenet1_1_path} --dataset=shapes_2d --output-dir={analysis_dir}",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed rendering of prototypes" in stdout


def test_local_analysis(analysis_dir, shapes_2d_squeezenet1_1_path):
    stdout, _ = python(
        f"-u -m protopnet viz local --model-path={shapes_2d_squeezenet1_1_path} --dataset=shapes_2d --sample=6 --output-dir={analysis_dir}",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed local analysis. Saved 6 analyses." in stdout


@pytest.mark.xfail(reason="This test is failing due to a bug in the code")
def test_pacmap(analysis_dir, shapes_2d_squeezenet1_1_path):
    stdout, _ = python(
        f"-u -m protopnet pacmap --model-path={shapes_2d_squeezenet1_1_path} --dataset=shapes_2d --n-neighbors=1 --save-dir={analysis_dir} --sample=1",
        {"WANDB_MODE": "dryrun"},
    )
    assert "PaCMAP plotted" in stdout

    pacmap_path = pathlib.Path(analysis_dir / "pacmap.png")
    assert pacmap_path.exists()

    pacmap_image = Image.open(pacmap_path)
    assert isinstance(pacmap_image, Image.Image)
    assert pacmap_image.format == "PNG"
    assert pacmap_image.mode == "RGBA"

    pacmap_image.load()


def test_global_analysis(analysis_dir, shapes_2d_squeezenet1_1_path):
    stdout, _ = python(
        f"-u -m protopnet viz global --model-path={shapes_2d_squeezenet1_1_path} --dataset=shapes_2d --output-dir={analysis_dir}",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed global analysis." in stdout
