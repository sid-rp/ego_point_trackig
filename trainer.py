import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
from datasets import EpicKitchenDataset,EgoPoints
from models import DinoDeltaModel,CrocoDeltaNet
from utils.losses import * 
from utils import misc
from collections import defaultdict
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler



def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))  # A large window size to smooth the values
    header = f'Train Epoch: [{epoch}]'

    for batch_idx, batch in enumerate(metric_logger.log_every(dataloader, args.print_freq, header)):
        if batch is None:
            continue
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        kp1 = batch['kp1'].to(device)
        kp2 = batch['kp2'].to(device)
        kp1_mask  = batch["kp1_mask"].to(device)
        kp2_mask = batch["kp2_mask"].to(device)
        # F_matrix  = batch["F_matrix"].to(device)
        # img1_dino_feat = batch["img1_dino_feat"].to(device)
        # img2_dino_feat = batch["img2_dino_feat"].to(device)
        
   
        optimizer.zero_grad()
       
        with torch.cuda.amp.autocast(enabled=True):
           
            feat1, feat2 = model(img1, img2)
            loss = criterion(feat1, feat2, kp1, kp2 ,kp1_mask, kp2_mask)

        loss_value = loss.item()  # Get the value of loss to log
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metric logger with the current loss
        metric_logger.update(loss=loss_value)
        # Log loss to tensorboard if necessary
        if writer is not None and batch_idx % args.write_freq == 0:
            writer.add_scalar('Loss/Train', loss_value, epoch * len(dataloader) + batch_idx)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR/Train', current_lr, epoch * len(dataloader) + batch_idx)

    # Average the loss over all batches
    avg_loss = metric_logger.meters['loss'].global_avg
    writer.add_scalar('Loss/Train_avg', avg_loss, epoch)
    print(f"Epoch [{epoch+1}], Train Loss: {avg_loss:.4f}")

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch, writer, args, prefix='val'):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))  # A large window size to smooth the values
    header = f'Validation Epoch: [{epoch}]'

    for batch_idx, batch in enumerate(metric_logger.log_every(dataloader, args.print_freq, header)):
        if batch is None:
            continue
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        kp1 = batch['kp1'].to(device)
        kp2 = batch['kp2'].to(device)

        kp1_mask  = batch["kp1_mask"].to(device)
        kp2_mask = batch["kp2_mask"].to(device)
        
        with torch.cuda.amp.autocast(enabled=True):
            feat1, feat2 = model(img1, img2)
            loss = criterion(feat1, feat2, kp1, kp2 ,kp1_mask, kp2_mask)

        loss_value = loss.item()  # Get the value of loss to log

        metric_logger.update(loss=loss_value)

        if writer is not None and batch_idx % args.write_freq == 0:
            writer.add_scalar(f'Loss/{prefix}', loss_value, epoch * len(dataloader) + batch_idx)

    metric_logger.synchronize_between_processes()

    avg_loss = metric_logger.meters['loss'].global_avg
    writer.add_scalar(f'Loss/{prefix}_avg', avg_loss, epoch)
    print(f"Epoch [{epoch+1}], {prefix.capitalize()} Loss: {avg_loss:.4f}")

    return avg_loss  # Return the average validation loss

def get_subsampled_dataloader(dataset, batch_size, num_workers, subsample_size, collate_fn=None):
    """
    Returns a DataLoader with a randomly subsampled dataset for one epoch.

    Args:
        dataset (Dataset): The full dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        subsample_size (int): Number of samples to randomly select.
        collate_fn (callable, optional): Custom collate function.

    Returns:
        DataLoader: A DataLoader for the randomly selected subset.
    """
    total_samples = len(dataset)
    subsample_size = min(subsample_size, total_samples)
    subsample_indices = torch.randperm(total_samples)[:subsample_size]
    sampler = SubsetRandomSampler(subsample_indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

def get_args():
    parser = argparse.ArgumentParser(description="Train the DinoDelta model on Epic Kitchen dataset")
    
    # Dataset arguments
    parser.add_argument('--root', type=str, default='/scratch/projects/fouheylab/shared_datasets/point_tracking_data/', help="Root directory for datasets")
    parser.add_argument('--overlap_threshold', type=float, default=0.4, help="Threshold for camera frustum overlap")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and validation")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--print_freq', type=int, default=10, help="Frequency of printing training stats")
    parser.add_argument('--write_freq', type=int, default=100, help="Frequency of writing stats to tensorboard")
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Number of warm-up epochs to increase learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the model checkpoints")

    # Model arguments
    parser.add_argument('--loss_function', type=str, default="l2",  help="loss function for optimization (e.g l2, cosine)")
    parser.add_argument('--arch', type=str, default="croco",  help="Archicture (e.g croco, dino)")
    parser.add_argument('--use_delta', type=bool, default=False,  help="Use delta net or not")


    parser.add_argument('--learning_rate', type=float, default=1e-6, help="Learning rate for the optimizer")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to run the model on")

    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, args):
    """Adjusts the learning rate with warm-up followed by cosine decay, ensuring it doesn't go below min_lr."""
    
    if epoch < args.warmup_epochs:
        # Linear warm-up
        lr = args.learning_rate * (epoch + 1) / args.warmup_epochs
    else:
        # Cosine decay after warm-up
        lr = args.min_lr + (args.learning_rate - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)))
    
    # Clamp to min_lr just in case
    lr = max(lr, args.min_lr)

    # Apply to all parameter groups
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr  # Useful if you want to log the current LR
    # print(f"Epoch {epoch+1}: Learning rate is {lr:.8f}")  # Print the learning rate for monitoring

def trainer(args):
    def worker_init_fn(worker_id):
        dataset = torch.utils.data.get_worker_info().dataset
        dataset._load_and_cache_all_tar_files()
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    scaler = GradScaler()  # For safe gradient scaling
    # Initialize model and optimizer
    if args.arch == "dino":
        model = DinoDeltaModel(delta=args.use_delta).to(device)
    elif args.arch == "croco":
         model  = CrocoDeltaNet(delta=args.use_delta).to(device)

    def get_param_groups(model, weight_decay=0.05):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if name.endswith("bias") or "bn" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    print(model)
    param_groups = get_param_groups(model, weight_decay=0.05)

    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    print(optimizer)



    os.makedirs(args.output_dir, exist_ok=True)
    # TensorBoard writer: Save logs to output directory
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Initialize best validation loss
    best_val_loss = float('inf')
    if args.loss_function == "l2":
        criterion = KeypointAlignmentLossL2()
    elif args.loss_function == "cosine":
        criterion = KeypointAlignmentLossCosineSimilarity()
    elif args.loss_function == "combined":
        criterion = CombinedKeypointLoss()

        
    # Create datasets
    train_dataset = EgoPoints(root=args.root, split = "train")
    val_dataset = EgoPoints(root=args.root, split = "val")
    train_dataset.set_epoch(0, subset_size=10_000)  # Random 9000 samples for epoch 0
    val_dataset.set_epoch(0, subset_size=2_000)


    train_dataloader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=misc.custom_collate_fn,
        persistent_workers=True
    )

    val_dataloader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=misc.custom_collate_fn,
        persistent_workers=True
    )
    # Training and validation loop
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)  # Adjust learning rate based on epoch
        if epoch>0:
                train_dataloader.dataset.set_epoch(epoch, subset_size=10_000)
                val_dataloader.dataset.set_epoch(epoch, subset_size=2_000)
        train_one_epoch(model, train_dataloader, optimizer, criterion, device, epoch, writer, scaler, args)
        avg_val_loss = validate_one_epoch(model, val_dataloader,criterion, device, epoch, writer, args, prefix='val')

        # Save the model if validation loss is improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(args.output_dir, f"model_loss_{args.loss_function}_epoch_{epoch+1}_val_loss_{best_val_loss:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} with validation loss {best_val_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    args = get_args()  # Get command-line arguments
    trainer(args)  # Call the trainer function to start training and validation
