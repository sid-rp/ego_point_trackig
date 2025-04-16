import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
from datasets import EpicKitchenDataset
from models import DinoDeltaModel
from utils.losses import * 
from utils import misc
from collections import defaultdict
import argparse
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, scaler, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))  # A large window size to smooth the values
    header = f'Train Epoch: [{epoch}]'

    for batch_idx, batch in enumerate(metric_logger.log_every(dataloader, args.print_freq, header)):
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
        if args.displacement_net:
            with torch.cuda.amp.autocast(enabled=True):
                feat1, feat2, delta, pixels_displacement = model(img1, img2)
                feat_keypoint_loss = criterion(feat1, feat2, kp1, kp2 ,kp1_mask, kp2_mask,  delta)
                pixel_loss  = PixelDisplacementLoss()(kp1,kp2,kp1_mask, kp2_mask, pixels_displacement)
                loss  = feat_keypoint_loss + 0.002 * pixel_loss
        else:
            with torch.cuda.amp.autocast(enabled=True):
                feat1, feat2, delta = model(img1, img2)
                loss = criterion(feat1, feat2, kp1, kp2 ,kp1_mask, kp2_mask,  delta)

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
        img1 = batch['img1'].to(device)
        img2 = batch['img2'].to(device)
        kp1 = batch['kp1'].to(device)
        kp2 = batch['kp2'].to(device)

        kp1_mask  = batch["kp1_mask"].to(device)
        kp2_mask = batch["kp2_mask"].to(device)
        # F_matrix  = batch["F_matrix"].to(device)
        # img1_dino_feat = batch["img1_dino_feat"].to(device)
        # img2_dino_feat = batch["img2_dino_feat"].to(device)
        
        if args.displacement_net:
            with torch.cuda.amp.autocast(enabled=True):
                feat1, feat2, delta, pixels_displacement = model(img1, img2)
                feat_keypoint_loss = criterion(feat1, feat2, kp1, kp2 ,kp1_mask, kp2_mask,  delta)
                pixel_loss  = PixelDisplacementLoss()(kp1,kp2,kp1_mask, kp2_mask, pixels_displacement)
                loss  = feat_keypoint_loss + 0.0020 * pixel_loss
        else:
            with torch.cuda.amp.autocast(enabled=True):
                feat1, feat2, delta = model(img1, img2)
                loss = criterion(feat1, feat2, kp1, kp2 ,kp1_mask, kp2_mask,  delta)
           
        loss_value = loss.item()  # Get the value of loss to log

        metric_logger.update(loss=loss_value)

        if writer is not None and batch_idx % args.write_freq == 0:
            writer.add_scalar(f'Loss/{prefix}', loss_value, epoch * len(dataloader) + batch_idx)

    metric_logger.synchronize_between_processes()

    avg_loss = metric_logger.meters['loss'].global_avg
    writer.add_scalar(f'Loss/{prefix}_avg', avg_loss, epoch)
    print(f"Epoch [{epoch+1}], {prefix.capitalize()} Loss: {avg_loss:.4f}")

    return avg_loss  # Return the average validation loss

def get_args():
    parser = argparse.ArgumentParser(description="Train the DinoDelta model on Epic Kitchen dataset")
    
    # Dataset arguments
    parser.add_argument('--root', type=str, default='/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/', help="Root directory for datasets")
    parser.add_argument('--overlap_threshold', type=float, default=0.2, help="Threshold for camera frustum overlap")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and validation")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for DataLoader")

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--print_freq', type=int, default=10, help="Frequency of printing training stats")
    parser.add_argument('--write_freq', type=int, default=100, help="Frequency of writing stats to tensorboard")
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Number of warm-up epochs to increase learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the model checkpoints")

    # model arguments
    parser.add_argument('--displacement_net', type=bool, default = False, help="Flag to train displacement network")



    # Model arguments
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
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

    
    # Create datasets
    train_dataset = EpicKitchenDataset(root=args.root, overlap_threshold=args.overlap_threshold, split="train")
    val_dataset = EpicKitchenDataset(root=args.root, overlap_threshold=args.overlap_threshold, split="val")

    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = misc.custom_collate_fn, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn = misc.custom_collate_fn, worker_init_fn=worker_init_fn)

    # Initialize model and optimizer
    model = DinoDeltaModel(displacement_net = args.displacement_net).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard writer: Save logs to output directory
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Initialize best validation loss
    best_val_loss = float('inf')

    # Training and validation loop
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)  # Adjust learning rate based on epoch
        train_one_epoch(model, train_dataloader, optimizer, KeypointAlignmentLoss(), device, epoch, writer, scaler, args)
        avg_val_loss = validate_one_epoch(model, val_dataloader,KeypointAlignmentLoss(), device, epoch, writer, args, prefix='val')

        # Save the model if validation loss is improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}_val_loss_{best_val_loss:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} with validation loss {best_val_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    args = get_args()  # Get command-line arguments
    trainer(args)  # Call the trainer function to start training and validation
