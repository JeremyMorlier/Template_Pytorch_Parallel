import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import argparse

try:
    import hostlist
except ImportError:
    print(
        "hostlist not available. Install it with `pip install hostlist` if using SLURM."
    )

from utils import init_distributed_mode


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to train the model
def train(args):
    init_distributed_mode(args)

    # Create model and move to GPU
    model = SimpleNN().to(args.gpu)
    model = torch.compile(model)  # Compile the model
    model = DDP(model, device_ids=[args.gpu])  # Wrap with DDP

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(args.gpu), target.to(args.gpu)
            data = data.view(data.size(0), -1)  # Flatten the data
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and args.rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    # Clean up
    dist.destroy_process_group()


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
