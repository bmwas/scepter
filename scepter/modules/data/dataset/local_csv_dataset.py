# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
import torch
import os

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.registry import DATASETS

@DATASETS.register_class()
class CSVInRAMDataset(BaseDataset):
    """
    Dataset for loading image-text pairs from a CSV file.
    
    Args:
        cfg: Configuration with the following keys:
            CSV_PATH: Path to the CSV file
            KEY: Column name containing image paths
            VALUE: Column name containing caption text
    """
    para_dict = {
        'CSV_PATH': {
            'value': '',
            'description': 'Path to the CSV file'
        },
        'KEY': {
            'value': 'image_path',
            'description': 'Column name containing image paths'
        },
        'VALUE': {
            'value': 'caption',
            'description': 'Column name containing caption text'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CSVInRAMDataset, self).__init__(cfg, logger=logger)
        
        # Load CSV data
        csv_path = cfg.CSV_PATH
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.key = cfg.get('KEY', 'image_path')
        self.value = cfg.get('VALUE', 'caption')
        
        # Verify columns exist
        if self.key not in self.df.columns:
            raise ValueError(f"Column '{self.key}' not found in CSV")
        if self.value not in self.df.columns:
            raise ValueError(f"Column '{self.value}' not found in CSV")
            
        self.real_number = len(self.df)
        self.logger.info(f"Loaded {self.real_number} samples from {csv_path}")

    def _get(self, idx):
        """Get a single item from the dataset."""
        row = self.df.iloc[idx]
        
        # Load and process image
        img_path = row[self.key]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder or alternative
            img = Image.new("RGB", (512, 512), color="gray")
        
        # Get caption
        caption = row[self.value]
        
        # Return in the format expected by the model
        return {
            "meta": {"image_path": img_path},
            "prompt": caption,
            "image": img
        }
