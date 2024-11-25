import yaml
import socket
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from time import time, strftime, gmtime

import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler

from twix import TWiX
from data import DatasetTWiX
from loss import BidirectionalContrastiveLoss

torch.backends.cuda.enable_flash_sdp(enabled=True)  # Flash attention

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_parser():

    parser = argparse.ArgumentParser(description="Training the TWiX module")

    # Dataset
    parser.add_argument('--dataset', type=str, help="Multiple object tracking dataset")
    parser.add_argument('--subset_train', type=str, help="Subset for training")
    parser.add_argument('--subset_val', type=str, help="Subset for validation")

    # Characterics data
    parser.add_argument('--WP', type=str, help="Temporal window for past tracklets (in seconds)")
    parser.add_argument('--WF', type=str, default='1f', help="Temporal window for future tracklets (in frames)")
    parser.add_argument('--max_gap', type=float, help="Maximal temporal horizon (in seconds)")
    parser.add_argument('--tracklet_name', type=str, default='TrackerIoU')
    parser.add_argument('--strategy', type=str, help="Strategy to create the pairs of tracklets ('frame' or 'tracklet')")

    # Model
    parser.add_argument('--d_model', type=int, default=32, help="Dimension in the Transformer")
    parser.add_argument('--nhead', type=int, default=16, help="Number of attention heads")
    parser.add_argument('--dim_feedforward', type=int, default=32, help="Dimension of the feed forward layer")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout in the Transformer")
    parser.add_argument('--num_layers', type=int, help="Number of TransformerEncoder layers")
    parser.add_argument('--inter_pair', action='store_true', help="Inter-pair Transformer Encoder")

    # Training and optimization
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--n_epoch', type=int, default=30, help="Number of epoch")

    # Loss funtion
    parser.add_argument('--B', type=int, default=1024, help="Batch size for bdrC")
    parser.add_argument('--tau', type=float, default=0.1, help="tau for bdrc")

    # Debug
    parser.add_argument('--seed', type=int, default=46, help="Seed")
    parser.add_argument('--save_each', default=-1, type=int, help="Save each n epoch, -1 means never")
    parser.add_argument('--n_batches_val', type=int, default=1000, help="Number of batches to evaluate on the validation set")

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(dataloader_train, model, criterion, optimizer, epoch, tensorboard, scaler):
    
    model.train()
    list_losses = []
    for coordsP, framesP, coordsF, framesF, Y, dict_infos in tqdm(dataloader_train, total=len(dataloader_train), desc=f'Training at epoch {epoch}', leave=False):

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.autocast(device_type=DEVICE):
            output = model(coordsP, framesP, coordsF, framesF, fps=dict_infos['fps'])
            loss = criterion(output, Y)

        # Backward with FP16
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        list_losses.append(loss.item())

    # Log the loss
    avg_loss = np.mean(list_losses)
    tensorboard.add_scalar("Loss/Train", avg_loss, epoch)

    return avg_loss


def eval(data_loader, model, criterion, epoch, tensorboard):

    model.eval()
    list_losses = []
    for coordsP, framesP, coordsF, framesF, Y, dict_infos in tqdm(data_loader, total=len(data_loader), desc=f"Evaluation at epoch {epoch}", leave=False):

        with torch.no_grad():
            with torch.autocast(device_type=DEVICE):
                output = model(coordsP, framesP, coordsF, framesF, fps=dict_infos['fps'])
                loss = criterion(output, Y)

        list_losses.append(loss.item())

    # Overall Loss
    avg_loss = np.mean(list_losses)
    tensorboard.add_scalar("Loss/Valid", avg_loss, epoch)

    return avg_loss


def main(opts, tensorboard):
    
    # Training dataset
    training_data = DatasetTWiX.load(f"data/{opts.dataset}/{opts.tracklet_name}/data_{opts.subset_train}_{opts.strategy}_{opts.max_gap:0.2f}_{opts.WP}_{opts.WF}.pt")

    # Validation dataset
    validation_data = DatasetTWiX.load(f"data/{opts.dataset}/{opts.tracklet_name}/data_{opts.subset_val}_{opts.strategy}_{opts.max_gap:0.2f}_{opts.WP}_{opts.WF}.pt")

    # Model
    model = TWiX(d_model=opts.d_model, nhead=opts.nhead, dim_feedforward=opts.dim_feedforward, num_layers=opts.num_layers, dropout=opts.dropout, inter_pair=opts.inter_pair, device=DEVICE)
    model.to(DEVICE)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters}")

    # Define loss function, optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    criterion = BidirectionalContrastiveLoss(tau=opts.tau, B=opts.B, device=DEVICE)
    scheduler = CosineAnnealingLR(optimizer, opts.n_epoch, eta_min=0)

    # Create a DataLoader for batching and shuffle the data for training and validation
    dataloader_train = DataLoader(training_data, batch_size=None, shuffle=True)
    dataloader_val = DataLoader(validation_data, batch_size=None, sampler=SubsetRandomSampler(range(min(len(validation_data), opts.n_batches_val))))

    # FP16
    scaler = GradScaler()

    # Training loop
    best_loss_val = np.inf
    start_train = time()
    for epoch in range(1, opts.n_epoch+1):

        # Train on one epoch
        train_epoch_loss = train(dataloader_train, model, criterion, optimizer, epoch, tensorboard, scaler)

        # Save the checkpoints
        if (opts.save_each != -1) and ((epoch % opts.save_each == 0) or (epoch == opts.n_epoch)):
            torch.save({'state_dict': model.state_dict()}, Path(f"{folderbase}/weights_{epoch}.pth.tar"))

        val_epoch_loss = eval(dataloader_val, model, criterion, epoch, tensorboard)

        print(f"{strftime('%H:%M:%S', gmtime(time() - start_train))} Epoch {epoch:02d}/{opts.n_epoch}\tTraining Loss={train_epoch_loss:.4f}\tValidation Loss={val_epoch_loss:.4f}")

        # Learning rate scheduler
        scheduler.step()

        # Save the best model according to the validation loss
        if val_epoch_loss <= best_loss_val:
            torch.save({'state_dict': model.state_dict()}, Path(f"{folderbase}/best_weights.pth.tar"))
        best_loss_val = min(best_loss_val, val_epoch_loss)


if __name__ == '__main__':

    # Get the arguments from CLI
    args = get_parser().parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)

    # Tensorboard
    results_folder = 'rlta' if args.strategy == 'tracklet' else 'rsta'  # for tracklet (LTA) or frame (STA)
    date_str = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"   # Name of the logging
    print(f"Name of the experiment: {date_str}_{socket.gethostname()}")
    folderbase = Path(f"{results_folder}/{args.dataset}/{date_str}_{socket.gethostname()}")
    tb = SummaryWriter(log_dir=folderbase)

    # Log the arguments
    with open(f'{folderbase}/args.json', 'w') as f:
        yaml.dump(args.__dict__, f)
    
    # Run the training process
    torch.cuda.empty_cache()  # VRAM usage
    tic = time()
    main(args, tb)
    print(f'Execution time: {strftime("%H:%M:%S", gmtime(time() - tic))}')
    print(f"Max VRAM used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    # Close Tensorboard 
    tb.close()