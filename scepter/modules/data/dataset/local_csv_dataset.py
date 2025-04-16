# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import sys

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
        },
        'ADD_INDICATOR': {
            'value': False,
            'description': 'Add {image} indicator to prompt if not present'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CSVInRAMDataset, self).__init__(cfg, logger=logger)
        
        # Load configuration
        csv_path = cfg.CSV_PATH
        self.mode = cfg.get('MODE', 'train')
        self.prompt_prefix = cfg.get('PROMPT_PREFIX', '')
        self.max_seq_len = cfg.get('MAX_SEQ_LEN', 1024)
        self.add_indicator = cfg.get('ADD_INDICATOR', False)
        
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

    def __len__(self):
        """Return the number of items in the dataset."""
        if self.mode == 'train':
            # We're using LoopSampler for training, so return a reasonable size
            # instead of sys.maxsize to avoid overflow with default samplers
            return self.real_number
        else:
            return self.real_number

    def _get(self, idx):
        """Get a single item from the dataset."""
        # For infinite dataset, wrap around the index
        row = self.df.iloc[idx % self.real_number]
        
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
        
        # Create masks (ones tensors with same spatial dimensions as images)
        src_mask = torch.ones_like(source_img[[0]])  # Take first channel and make mask
        tar_mask = torch.ones_like(target_img[[0]])
        
        # Get caption/prompt
        prompt = row['Prompt']
        if self.prompt_prefix:
            prompt = f"{self.prompt_prefix} {prompt}"
            
        # Add image indicator if needed
        if self.add_indicator:
            if '{image}' not in prompt:
                prompt = '{image}, ' + prompt
        
        # Return in the format expected by the ACE model
        return {
            'src_image_list': [source_img],  # List of images
            'src_mask_list': [src_mask],     # List of masks
            'image': target_img,             # Target image
            'image_mask': tar_mask,          # Target mask
            'prompt': [[prompt]],            # List of list of prompts
            'edit_id': [0]                   # Edit IDs
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to properly batch items.
        This ensures items are properly stacked for the model.
        """
        batch_dict = {}
        for k in batch[0].keys():
            if k in ['src_image_list', 'src_mask_list', 'edit_id']:
                # For list fields, we need to concatenate them
                batch_dict[k] = sum([item[k] for item in batch], [])
            elif k == 'prompt':
                # For prompt, keep as list of lists
                batch_dict[k] = [item['prompt'][0] for item in batch]
            elif k in ['image', 'image_mask']:
                # For tensor fields, we stack them
                batch_dict[k] = torch.stack([item[k] for item in batch])
            else:
                # Other fields, just collect them
                batch_dict[k] = [item[k] for item in batch]
        return batch_dict
