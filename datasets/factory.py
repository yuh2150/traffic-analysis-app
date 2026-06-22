import logging
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional
from .dataset import TrafficDataset, CocoDataset

logger = logging.getLogger("DatasetFactory")


class DatasetFactory:
    """Factory to instantiate dataset loaders for training and validation."""

    @staticmethod
    def get_dataloader(
        img_dir: str,
        anno_dir: str,
        batch_size: int = 2,
        img_size: Tuple[int, int] = (640, 640),
        shuffle: bool = False,
        max_samples: Optional[int] = None,
        dataset_type: str = "detrac",
    ) -> DataLoader:
        """Instantiates and returns a DataLoader for the selected dataset.

        Args:
            img_dir: Folder path containing images.
            anno_dir: Folder path containing annotations (XML folder for DETRAC, JSON file for COCO).
            batch_size: Loader batch size.
            img_size: Image resolution (H, W).
            shuffle: Whether to shuffle data.
            max_samples: If set, subsets the dataset for speed.
            dataset_type: Dataset format type ('detrac' or 'coco').
        """
        logger.info(f"Initializing {dataset_type.upper()} dataset loader from: {img_dir}")
        if dataset_type.lower() == "coco":
            # For COCO, anno_dir is the path to the annotation JSON file
            dataset = CocoDataset(img_dir=img_dir, anno_file=anno_dir, img_size=img_size, max_samples=max_samples)
        else:
            dataset = TrafficDataset(img_dir=img_dir, anno_dir=anno_dir, img_size=img_size)
            # Apply dataset subsetting if max_samples is requested for DETRAC
            if max_samples is not None and 0 < max_samples < len(dataset):
                logger.info(f"Subsetting dataset to the first {max_samples} samples.")
                indices = list(range(max_samples))
                dataset = Subset(dataset, indices)

        def collate_fn(batch):
            return tuple(zip(*batch))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn
        )

