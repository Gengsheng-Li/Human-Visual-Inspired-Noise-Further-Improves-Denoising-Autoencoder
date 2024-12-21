import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from denoise_autoencoder import DenoisingAutoencoder, parse_args
from utils import create_run_name

class LinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(LinearProbe, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(128 * 12 * 12, num_classes)  # Adjust size based on your encoder output

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

def test_linear_probe(args):
    # Create a unique run name
    run_name = create_run_name(args)

    # Initialize wandb with custom name
    wandb.init(project=args.project_name, name=run_name, config=args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained autoencoder
    autoencoder = DenoisingAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(args.model_save_path))

    # Create linear probe model
    model = LinearProbe(autoencoder.encoder).to(device)

    # Load STL-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.STL10(root=args.data_path, split='train', download=True, transform=transform)
    test_dataset = datasets.STL10(root=args.data_path, split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)

        print(f'\nEpoch {epoch}:')
        print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {train_correct}/{len(train_loader.dataset)} ({train_accuracy:.2f})')
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_correct}/{len(test_loader.dataset)} ({test_accuracy:.2f})\n')

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

    wandb.finish()

if __name__ == '__main__':
    args = parse_args()  # Assuming you've defined parse_args() as before
    test_linear_probe(args)