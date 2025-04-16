# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.registry import DATASETS
from scepter.modules.utils.file_system import FS

@DATASETS.register_class()
class CSVInRAMDataset(BaseDataset):
    """
    Dataset for loading image-text pairs from a CSV file.
    
    Args:
        cfg: Configuration with the following keys:
            CSV_PATH: Path to the CSV file
            MODE: Dataset mode (train, validation, etc.)
    """
    para_dict = {
        'CSV_PATH': {
            'value': '',
            'description': 'Path to the CSV file'
        },
        'MODE': {
            'value': 'train',
            'description': 'Dataset mode (train, validation, etc.)'
        },
        'PROMPT_PREFIX': {
            'value': '',
            'description': 'Prefix to add to all prompts'
        },
        'MAX_SEQ_LEN': {
            'value': 1024,
            'description': 'Maximum sequence length for text'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CSVInRAMDataset, self).__init__(cfg, logger=logger)
        
        # Load configuration
        csv_path = cfg.CSV_PATH
        self.mode = cfg.get('MODE', 'train')
        self.prompt_prefix = cfg.get('PROMPT_PREFIX', '')
        self.max_seq_len = cfg.get('MAX_SEQ_LEN', 1024)
        
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Load CSV data - using comma as delimiter
        self.df = pd.read_csv(csv_path)
        
        # Check for required columns
        required_columns = ['Source:FILE', 'Prompt']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
        self.real_number = len(self.df)
        self.logger.info(f"Loaded {self.real_number} samples from {csv_path}")
        
        # Setup transforms
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _get(self, idx):
        """Get a single item from the dataset."""
        row = self.df.iloc[idx]
        
        # Load source image (used for input)
        source_path = row['Source:FILE']
        try:
            source_img = Image.open(source_path).convert("RGB")
            source_img = self.transforms(source_img)
        except Exception as e:
            self.logger.error(f"Error loading source image {source_path}: {e}")
            # Return a placeholder
            source_img = torch.zeros((3, 512, 512))
        
        # Use source as target since Target:FILE is not available
        target_img = source_img
        
        # Get caption/prompt
        prompt = row['Prompt']
        if self.prompt_prefix:
            prompt = f"{self.prompt_prefix} {prompt}"
        
        # Return in the format expected by the model
        return {
            "source": source_img,
            "target": target_img,
            "meta": {
                "source_path": source_path,
                "target_path": source_path  # Use source path as target path
            },
            "prompt": prompt
        }
