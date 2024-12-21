# import os
# os.environ["WANDB_DISABLED"]="true"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import wandb
from utils import simulate_human_vision
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a denoising autoencoder')
    parser.add_argument('--data_path', type=str, default='E:\Dataset\STL10', help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of worker processes for data loading (default: 4)')
    parser.add_argument('--project_name', type=str, default='human_visual_inspired_denoised_autoencoder', help='project name for wandb')
    parser.add_argument('--model_save_path', type=str, default='denoising_autoencoder.pth', help='path to save the trained model')
    
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches(steps) to wait before logging training status')
    parser.add_argument('--visualize_interval', type=int, default=10, help='how many epochs to wait before visualizing reconstruction')
    parser.add_argument('--specific_indices', type=int, nargs='+', default=[5, 10, 15, 20], help='specific indices for tracking reconstruction')
    
    return parser.parse_args()

# Define the autoencoder architecture
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Custom dataset with visual effects
class VisualEffectsDataset(datasets.STL10):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Apply visual effects
        noisy_img = simulate_human_vision(
            img_np,
            # noise_params={'center_noise_level': 0.1, 'edge_noise_level': 0.5},
            # color_params={'color_fade_factor': 0.7},
            # motion_params={'kernel_size_factor': 0.1, 'angle': 45},
            mae_params_pixel={'mask_ratio': 0.75, 'mask_value': 0},
            # mae_params_patch={'patch_size': 6, 'mask_ratio': 0.75, 'mask_value': 0},
        )
        
        return transforms.ToTensor()(noisy_img), img, label

def visualize_reconstruction(model, dataset, device, epoch, specific_indices=None):
    model.eval()
    with torch.no_grad():
        if specific_indices is not None:
            # Get specific images
            specific_dataset = Subset(dataset, specific_indices)
            specific_loader = DataLoader(specific_dataset, batch_size=len(specific_indices), shuffle=False)
            noisy_imgs, clean_imgs, _ = next(iter(specific_loader))
        else:
            # Get a random batch of images
            random_loader = DataLoader(dataset, batch_size=4, shuffle=True)
            noisy_imgs, clean_imgs, _ = next(iter(random_loader))
        
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
        
        # Reconstruct images
        reconstructed = model(noisy_imgs)
        
        # Convert tensors to numpy arrays for plotting
        noisy_imgs = noisy_imgs.cpu().numpy()
        clean_imgs = clean_imgs.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, len(noisy_imgs), figsize=(4*len(noisy_imgs), 10))
        fig.suptitle(f'Epoch {epoch}: Noisy vs Clean vs Reconstructed')
        
        for i in range(len(noisy_imgs)):
            axes[0, i].imshow(np.transpose(noisy_imgs[i], (1, 2, 0)))
            axes[0, i].set_title('Noisy')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(np.transpose(clean_imgs[i], (1, 2, 0)))
            axes[1, i].set_title('Clean')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
            axes[2, i].set_title('Reconstructed')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Save the figure and log it to wandb
        image_type = "specific" if specific_indices is not None else "random"
        plt.savefig(f'reconstruction_{image_type}_epoch_{epoch}.png')
        wandb.log({f"reconstruction_{image_type}_epoch_{epoch}": wandb.Image(f'reconstruction_{image_type}_epoch_{epoch}.png')})
        plt.close()

def train(args):
    # Initialize wandb
    wandb.init(project=args.project_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = VisualEffectsDataset(root=args.data_path, split='unlabeled', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = VisualEffectsDataset(root=args.data_path, split='train', download=True, transform=transform)
    
    # Specific indices for tracking
    specific_indices = args.specific_indices
    
    # Initialize the model, loss function, and optimizer
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (noisy_imgs, clean_imgs, _) in enumerate(train_loader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        
        # Visualize reconstruction every visualize_interval epochs
        if (epoch + 1) % args.visualize_interval == 0:
            visualize_reconstruction(model, test_dataset, device, epoch + 1)
            visualize_reconstruction(model, test_dataset, device, epoch + 1, specific_indices)

    # Save the model
    torch.save(model.state_dict(), args.model_save_path)
    wandb.save(args.model_save_path)

    # Finish the wandb run
    wandb.finish()

    print("Training completed and model saved.")
    
if __name__ == '__main__':
    args = parse_args()
    train(args)