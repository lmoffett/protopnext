import os

import numpy as np
import pytest
import torch

from protopnet.datasets.shapes_2d import (
    ShapesDataset,
    ShapesSplitDataloaders,
    create_sample,
    generate_polka_dot_circle_texture,
    generate_striped_square_texture,
    train_dataloaders,
)


def test_generate_striped_square_texture():
    size = 20
    texture = generate_striped_square_texture(size)
    # verify the shape of the texture
    assert texture.shape == (size, size, 3)
    # check that the first stripe columns 0 to 4 is green and the next stripe is white
    stripe_width = 5
    for col in range(0, stripe_width):
        np.testing.assert_array_equal(
            texture[0, col], np.array([0, 255, 0], dtype=np.uint8)
        )
    for col in range(stripe_width, 2 * stripe_width):
        np.testing.assert_array_equal(
            texture[0, col], np.array([255, 255, 255], dtype=np.uint8)
        )


def test_generate_polka_dot_circle_texture():
    size = 30
    texture = generate_polka_dot_circle_texture(size)
    # verify the shape of the texture
    assert texture.shape == (size, size, 3)
    # check that the base color is blue
    assert np.all(texture[0, 0] == np.array([0, 0, 255], dtype=np.uint8))
    # verify that one of the dot centers has a white pixel
    center = (size // 3, size // 3)
    assert np.all(
        texture[center[1], center[0]] == np.array([255, 255, 255], dtype=np.uint8)
    )


def test_create_sample_with_specific_shapes():
    # Test square generation
    np.random.seed(42)  # Set seed for reproducibility
    local_rng = np.random.default_rng(42)

    # Test square generation
    image_square, top_left_x, top_left_y, obj_size = create_sample(
        object_size=25, image_size=256, shape_type="square", seed=42
    )

    # Check that the generated image is of expected dimensions
    assert image_square.shape == (256, 256, 3)

    # Find the green pixels that are characteristic of squares
    # (should be at least one green pixel if square was generated)
    green_pixels = np.where(
        (image_square[:, :, 0] == 0)
        & (image_square[:, :, 1] == 255)
        & (image_square[:, :, 2] == 0)
    )
    assert len(green_pixels[0]) > 0, "Square shape should contain green stripes"

    # Test circle generation
    image_circle, top_left_x, top_left_y, obj_size = create_sample(
        object_size=25, image_size=256, shape_type="circle", seed=42
    )

    # Check for blue pixels that are characteristic of circles
    blue_pixels = np.where(
        (image_circle[:, :, 0] == 0)
        & (image_circle[:, :, 1] == 0)
        & (image_circle[:, :, 2] == 255)
    )
    assert len(blue_pixels[0]) > 0, "Circle shape should contain blue background"

    # Ensure they're different images
    assert not np.array_equal(
        image_square, image_circle
    ), "Square and circle should produce different images"


def test_shapes_dataset_alternating():
    # Create dataset with alternating shapes
    samples = 10
    dataset = ShapesDataset(num_samples=samples, seed=123)

    # Check that shapes alternate: even indices (0,2,4...) are squares, odd (1,3,5...) are circles
    for i in range(samples):
        sample = dataset[i]
        if i % 2 == 0:
            assert sample["target"] == 0, f"Index {i} should be a square (class 0)"
        else:
            assert sample["target"] == 1, f"Index {i} should be a circle (class 1)"

    # Verify class balance is exactly 50/50
    targets = [dataset[i]["target"] for i in range(samples)]
    num_squares = sum(1 for t in targets if t == 0)
    num_circles = sum(1 for t in targets if t == 1)

    assert num_squares == samples // 2 + (samples % 2), "Should have half squares"
    assert num_circles == samples // 2, "Should have half circles"
    assert num_squares + num_circles == samples, "Total should equal sample count"


def test_shapes_dataset():
    # Create dataset for training samples
    train_samples = 6
    dataset = ShapesDataset(num_samples=train_samples, seed=123)

    # Check dataset length
    assert len(dataset) == train_samples

    # Check consistency of samples
    sample1 = dataset[0]
    sample2 = dataset[0]  # Getting the same index again should return the same data

    assert sample1["target"] == sample2["target"]
    assert sample1["sample_id"] == sample2["sample_id"]
    torch.testing.assert_close(sample1["img"], sample2["img"])
    torch.testing.assert_close(
        sample1["sample_bounding_box"], sample2["sample_bounding_box"]
    )

    # Check validation dataset with different seed
    val_samples = 4
    val_dataset = ShapesDataset(num_samples=val_samples, seed=124)
    assert len(val_dataset) == val_samples

    # Check sample properties
    sample = dataset[0]
    assert isinstance(sample["img"], torch.Tensor)
    assert sample["img"].shape == (3, 256, 256)  # CHW format
    assert sample["target"] in [0, 1]  # 0 for square, 1 for circle
    assert isinstance(sample["sample_id"], str)
    assert sample["sample_bounding_box"].shape == (4,)  # [x1, y1, x2, y2]


def test_shapes_split_dataloaders():
    # Create dataloaders with minimal samples
    dataloaders = ShapesSplitDataloaders(
        num_classes=2,
        batch_sizes={"train": 2, "project": 2, "val": 2, "test": 2},
        image_size=(128, 128),
        train_samples=6,
        val_samples=4,
        test_samples=2,
    )

    # Check that all loaders were created
    assert dataloaders.train_loader is not None
    assert dataloaders.project_loader is not None
    assert dataloaders.val_loader is not None
    assert dataloaders.test_loader is not None

    # Check batch sizes
    assert dataloaders.train_loader.batch_size == 2
    assert dataloaders.project_loader.batch_size == 2
    assert dataloaders.val_loader.batch_size == 2
    assert dataloaders.test_loader.batch_size == 2

    # Check number of classes
    assert dataloaders.num_classes == 2

    # Check shapes of data from loaders
    train_batch = next(iter(dataloaders.train_loader))
    assert train_batch["img"].shape == (2, 3, 128, 128)
    assert train_batch["target"].shape == (2,)


def test_train_dataloaders():
    # Test the factory function
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PPNXT_SHAPES_DS_SEED", "42")
    monkeypatch.setenv("PPNXT_SEED", "43")  # For torch operations

    dataloaders = train_dataloaders(
        image_size=(64, 64),
        batch_sizes={"train": 4, "project": 4, "val": 4},
        train_samples=10,
        val_samples=5,
        object_size=20,
    )

    # Verify the returned object has all expected attributes
    assert hasattr(dataloaders, "train_loader")
    assert hasattr(dataloaders, "project_loader")
    assert hasattr(dataloaders, "val_loader")
    assert hasattr(dataloaders, "num_classes")

    # Check that the dataloaders produce output in the expected shape
    train_batch = next(iter(dataloaders.train_loader))
    assert train_batch["img"].shape == (4, 3, 64, 64)

    # Check that data generation with same seed is deterministic
    train_samples1 = [dataloaders.train_loader.dataset[i]["target"] for i in range(3)]

    # Create new dataloaders with same seed
    monkeypatch.setenv("PPNXT_SHAPES_DS_SEED", "42")
    dataloaders2 = train_dataloaders(
        image_size=(64, 64),
        batch_sizes={"train": 4, "project": 4, "val": 4},
        train_samples=10,
        val_samples=6,
        object_size=20,
    )

    train_samples2 = [dataloaders2.train_loader.dataset[i]["target"] for i in range(3)]
    assert train_samples1 == train_samples2  # Should get same targets with same seed

    monkeypatch.undo()
