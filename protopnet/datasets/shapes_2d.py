import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from protopnet.datasets.dataset_prep import SplitDistribution

from .torch_extensions import seed_worker_for_reproducability

log = logging.getLogger(__name__)

# --- Shape Generation Functions ---


def generate_striped_square_texture(size, stripe_width=5, rng=None):
    """
    Generates a texture of the given size with vertical stripes.
    Stripes alternate between green (RGB: (0, 255, 0)) and white (RGB: (255, 255, 255)).

    Args:
        size: Size of the texture
        stripe_width: Width of each stripe
        rng: Optional numpy random generator instance
    """
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    for col in range(size):
        stripe_index = col // stripe_width
        color = (0, 255, 0) if stripe_index % 2 == 0 else (255, 255, 255)
        texture[:, col] = color
    return texture


def generate_polka_dot_circle_texture(size, dot_radius=None, num_dots=2, rng=None):
    """
    Generates a texture for a circle.
    The base is blue (RGB: (0, 0, 255)) and white polka dots are drawn on it.

    Args:
        size: Size of the texture
        dot_radius: Radius of each dot
        num_dots: Number of dots to draw
        rng: Numpy random generator instance for random dot placement
    """
    texture = np.full((size, size, 3), (0, 0, 255), dtype=np.uint8)

    if dot_radius is None:
        dot_radius = max(1, size // 10)

    centers = [(size // 3, size // 3), (2 * size // 3, 2 * size // 3)]
    # If more centers are needed than provided, generate random positions
    if num_dots > len(centers):
        margin = dot_radius + 2  # Add margin to keep dots fully inside
        for _ in range(num_dots - len(centers)):
            x = (
                rng.integers(margin, size - margin)
                if rng
                else np.random.randint(margin, size - margin)
            )
            y = (
                rng.integers(margin, size - margin)
                if rng
                else np.random.randint(margin, size - margin)
            )
            centers.append((x, y))
    # Use only the required number of centers
    centers = centers[:num_dots]

    for center in centers:
        cv2.circle(texture, center, dot_radius, (255, 255, 255), thickness=-1)
    return texture


def create_sample(
    object_size=25,
    image_size=256,
    stripe_width=5,
    dot_radius=None,
    dot_count=2,
    shape_type=None,
    seed=None,
):
    """
    Generates one sample consisting of:
      - A background of pure black.
      - A single object (square or circle) with textured fill.
    """

    # Create a local RNG instance specific to this sample
    local_rng = (
        np.random.Generator(np.random.PCG64(seed))
        if seed is not None
        else np.random.default_rng()
    )

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Use the local RNG to make shape decisions
    max_coord = image_size - object_size
    top_left_x = local_rng.integers(0, max_coord + 1)
    top_left_y = local_rng.integers(0, max_coord + 1)

    if shape_type == "square":
        texture = generate_striped_square_texture(
            object_size, stripe_width=stripe_width, rng=local_rng
        )
        image[
            top_left_y : top_left_y + object_size, top_left_x : top_left_x + object_size
        ] = texture
    else:
        texture = generate_polka_dot_circle_texture(
            object_size, dot_radius=dot_radius, num_dots=dot_count, rng=local_rng
        )
        shape_mask = np.zeros((object_size, object_size), dtype=np.uint8)
        center = (object_size // 2, object_size // 2)
        radius = object_size // 2
        cv2.circle(shape_mask, center, radius, 1, thickness=-1)
        region = image[
            top_left_y : top_left_y + object_size, top_left_x : top_left_x + object_size
        ]
        bool_mask = shape_mask.astype(bool)
        region[bool_mask] = texture[bool_mask]

    return image, top_left_x, top_left_y, object_size


class ShapesDataset(Dataset):
    """
    Dataset that generates 2D shapes (squares and circles) on-the-fly with deterministic seeding.
    """

    def __init__(
        self,
        num_samples: int = 10,
        image_size: int = 256,
        object_size: int = 25,
        stripe_width: int = 5,
        dot_radius: int = None,
        dot_count: int = 2,
        transform=None,
        seed: int = None,
    ):
        """
        Args:
            num_samples: Number of samples in the dataset
            image_size: Size of the generated images
            object_size: Size of the objects in the images
            stripe_width: Width of stripes for square textures
            dot_radius: Radius of dots for circle textures
            dot_count: Number of dots for circle textures
            transform: Any transforms to apply to the generated images
            seed: Random seed for reproducibility
        """
        if seed is None:
            # Look for dedicated dataset seed environment variable
            seed = int(os.environ.get("PPNXT_SHAPES_DS_SEED", 1234))
            log.warning(f"PPNXT_SHAPES_DS_SEED not set, using default value: {seed}")

        self.num_samples = num_samples
        self.image_size = image_size
        self.object_size = object_size
        self.stripe_width = stripe_width
        self.dot_radius = dot_radius
        self.dot_count = dot_count
        self.transform = transform
        self.base_seed = seed
        self.indices = list(range(self.num_samples))

        # Create class mapping for labels
        self.class_map = {"square": 0, "circle": 1}
        self.classes = list(self.class_map.keys())

        # Pre-determine shapes for each index to ensure consistency
        self._shape_types = {}
        for i in range(self.num_samples):
            # Use the RNG to make a deterministic choice
            self._shape_types[i] = "square" if i % 2 == 0 else "circle"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use a deterministic seed for this sample based on the base seed
        sample_seed = self.base_seed + idx

        # Ensure this sample always has the same shape type
        shape_type = self._shape_types[idx]

        # Create the sample
        image, top_left_x, top_left_y, obj_size = create_sample(
            object_size=self.object_size,
            image_size=self.image_size,
            stripe_width=self.stripe_width,
            dot_radius=self.dot_radius,
            dot_count=self.dot_count,
            shape_type=shape_type,
            seed=sample_seed,
        )

        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        # Create bounding box information
        bbox = torch.tensor(
            [
                top_left_x / self.image_size,
                top_left_y / self.image_size,
                (top_left_x + obj_size) / self.image_size,
                (top_left_y + obj_size) / self.image_size,
            ]
        ).float()

        # Return the sample in the format expected by the rest of the system
        result = {
            "img": image,
            "target": self.class_map[shape_type],
            "sample_id": str(idx),
            "sample_bounding_box": bbox,
        }

        return result


@dataclass
class ShapesSplitDataloaders:
    """
    Creates train, validation and optional project dataloaders for in-memory shapes dataset.
    Similar interface to FilesystemSplitDataloaders but generates data on-the-fly.
    """

    num_classes: int
    batch_sizes: Dict[str, int]
    image_size: Tuple[int, int]
    train_samples: int = 10
    val_samples: int = 10
    test_samples: int = 10
    object_size: int = 25
    stripe_width: int = 5
    dot_radius: int = None
    dot_count: int = 2
    augment: bool = False
    train_distribution: SplitDistribution = field(init=False)
    train_loader: torch.utils.data.DataLoader = field(init=False)
    project_loader: torch.utils.data.DataLoader = field(init=False)
    val_loader: torch.utils.data.DataLoader = field(init=False)
    test_loader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        """
        Create train, project, and validation dataloaders for the shapes dataset.
        """
        # Get dedicated dataset seed, with warning if not set
        dataset_seed = os.environ.get("PPNXT_SHAPES_DS_SEED")
        if dataset_seed is None:
            dataset_seed = 1234
            log.warning(f"PPNXT_SHAPES_DS_SEED not set, using default: {dataset_seed}")
        else:
            dataset_seed = int(dataset_seed)

        # Get general seed for torch operations
        torch_seed = int(os.environ.get("PPNXT_SEED", 1234))
        generator = torch.Generator()
        generator.manual_seed(torch_seed)

        # Calculate mean and std on a sample batch for normalization
        temp_dataset = ShapesDataset(
            num_samples=self.train_samples,
            image_size=self.image_size[0],
            object_size=self.object_size,
            stripe_width=self.stripe_width,
            dot_radius=self.dot_radius,
            dot_count=self.dot_count,
            seed=dataset_seed,
        )

        # Calculate mean and std from sample images
        images = torch.stack([temp_dataset[i]["img"] for i in range(len(temp_dataset))])
        mean = images.mean(dim=(0, 2, 3))
        std = images.std(dim=(0, 2, 3))

        self.train_distribution = SplitDistribution(mean, std)

        normalize = transforms.Normalize(mean=mean, std=std)

        # Create train dataset with augmentation if enabled
        train_transforms = []

        if self.augment:
            train_transforms = [
                transforms.RandomOrder(
                    [
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                        transforms.ColorJitter(
                            (0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(
                            degrees=10, shear=(-2, 2), translate=[0.05, 0.05]
                        ),
                    ]
                ),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=15),
                        transforms.RandomPerspective(distortion_scale=0.2),
                        transforms.RandomAffine(degrees=0, shear=10),
                    ]
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                normalize,
            ]

        # Create separate datasets for train/validation with different seeds
        train_dataset = ShapesDataset(
            num_samples=self.train_samples,
            image_size=self.image_size[0],
            object_size=self.object_size,
            stripe_width=self.stripe_width,
            dot_radius=self.dot_radius,
            dot_count=self.dot_count,
            transform=transforms.Compose(train_transforms),
            seed=dataset_seed,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_sizes["train"],
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            prefetch_factor=8,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )

        # Create project dataset (same as train but no augmentation)
        project_dataset = ShapesDataset(
            num_samples=self.train_samples,
            image_size=self.image_size[0],
            object_size=self.object_size,
            stripe_width=self.stripe_width,
            dot_radius=self.dot_radius,
            dot_count=self.dot_count,
            transform=normalize,
            seed=dataset_seed,
        )

        self.project_loader = DataLoader(
            project_dataset,
            batch_size=self.batch_sizes["project"],
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            prefetch_factor=8,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )

        # Create validation dataset with a different seed to ensure independence
        val_dataset = ShapesDataset(
            num_samples=self.val_samples,
            image_size=self.image_size[0],
            object_size=self.object_size,
            stripe_width=self.stripe_width,
            dot_radius=self.dot_radius,
            dot_count=self.dot_count,
            transform=normalize,
            seed=dataset_seed + 1,  # Different seed for validation
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_sizes["val"],
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            prefetch_factor=8,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )

        if self.test_samples != 0 and "test" in self.batch_sizes:

            test_dataset = ShapesDataset(
                num_samples=self.val_samples,
                image_size=self.image_size[0],
                object_size=self.object_size,
                stripe_width=self.stripe_width,
                dot_radius=self.dot_radius,
                dot_count=self.dot_count,
                transform=normalize,
                seed=dataset_seed + 2,  # Different seed for test
            )

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_sizes["test"],
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                prefetch_factor=8,
                generator=generator,
                worker_init_fn=seed_worker_for_reproducability,
            )


def train_dataloaders(
    data_path: Union[str, Path] = None,
    meta_data_path: Union[str, Path] = None,
    train_dir: str = None,
    val_dir: str = None,
    project_dir: str = None,
    image_size: Tuple[int, int] = (256, 256),
    batch_sizes: Dict[str, int] = {"train": 1, "project": 1, "val": 1, "test": 1},
    train_samples: int = 10,
    val_samples: int = 10,
    test_samples: int = 10,
    object_size: int = 25,
    stripe_width: int = 5,
    dot_radius: int = None,
    dot_count: int = 2,
):
    """
    Creates train, validation, and optional project dataloaders for the dataset.
    Generates the shapes dataset in memory using pseudorandom numbers.
    """
    # Get dedicated dataset seed
    dataset_seed = os.environ.get("PPNXT_SHAPES_DS_SEED")
    if dataset_seed is None:
        dataset_seed = 1234
        log.warning(f"PPNXT_SHAPES_DS_SEED not set, using default: {dataset_seed}")
    else:
        dataset_seed = int(dataset_seed)

    log.info(f"Creating in-memory shapes dataset with seed {dataset_seed}")

    return ShapesSplitDataloaders(
        num_classes=2,
        batch_sizes=batch_sizes,
        image_size=image_size,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        object_size=object_size,
        stripe_width=stripe_width,
        dot_radius=dot_radius,
        dot_count=dot_count,
    )
