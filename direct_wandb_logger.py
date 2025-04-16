#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct WandB Loss Logger

This script monitors the training log output and sends loss values directly 
to Weights & Biases. Run it alongside your training script.

Usage:
    python direct_wandb_logger.py <path_to_log_file>
    
    # Or to capture live output:
    python train.py | tee training.log | python direct_wandb_logger.py -
"""

import sys
import re
import time
import wandb
import argparse

# Pattern to match loss values in the log output
LOSS_PATTERN = re.compile(r'loss: (\d+\.\d+)\((\d+\.\d+)\)')

def main():
    parser = argparse.ArgumentParser(description='Monitor training logs and send loss values to WandB')
    parser.add_argument('log_file', help='Path to log file or "-" for stdin')
    parser.add_argument('--project', default='scepter-project', help='WandB project name')
    parser.add_argument('--run-name', default='direct-loss-tracking', help='WandB run name')
    parser.add_argument('--tags', default='loss-tracking', help='Comma-separated WandB tags')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.project,
        name=args.run_name,
        tags=args.tags.split(','),
        config={
            "source": "direct_logger",
            "timestamp": time.time()
        }
    )
    
    print(f"WandB initialized. Dashboard URL: {wandb.run.url}")
    print(f"Monitoring for loss values...")
    
    # Track step/iteration
    step = 0
    
    # Open input source
    if args.log_file == '-':
        f = sys.stdin
        print("Reading from stdin (piped output)...")
    else:
        f = open(args.log_file, 'r')
        print(f"Reading from file: {args.log_file}")
    
    try:
        # Keep reading lines
        while True:
            line = f.readline()
            
            # EOF
            if not line:
                if args.log_file != '-':
                    # For files, wait and try again
                    time.sleep(1)
                    continue
                else:
                    # For stdin, exit when done
                    break
            
            # Check for loss values
            match = LOSS_PATTERN.search(line)
            if match:
                current_loss = float(match.group(1))
                avg_loss = float(match.group(2))
                
                print(f"Found loss: current={current_loss}, avg={avg_loss}")
                
                # Log to wandb
                wandb.log({
                    "current_loss": current_loss,
                    "average_loss": avg_loss,
                    "step": step
                }, step=step)
                
                step += 1
    
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        if args.log_file != '-':
            f.close()
        
        # Close wandb run
        wandb.finish()
        print("WandB run complete.")

if __name__ == "__main__":
    main()
