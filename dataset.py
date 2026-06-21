import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure images are 32x32 pixels
    transforms.Grayscale(num_output_channels=3),  # Ensure images are grayscale
    transforms.RandomRotation(degrees=15),  # Data augmentation: random rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load MNIST training dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Load MNIST test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print("Sample:", train_dataset[0][0][0][0])  # Display a sample from the training dataset
    
    # Display a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
