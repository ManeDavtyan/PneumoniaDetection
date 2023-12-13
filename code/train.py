#Contains the training codes.
# It must accept a dataset path and hyperparameters as inputs.
# It should produce and save at least one checkpoint as output.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from code.model import VGG16
from code.dataset import PneumoniaDataset

def train_model(dataset_path, output_checkpoint_path, num_epochs=10, learning_rate=0.001, batch_size=32):
    # Set up your dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = PneumoniaDataset(dataset_path, transform, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the VGG16 model
    vgg16_model = VGG16()

    # Set up your loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(vgg16_model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        vgg16_model.train()

        for images, labels in train_dataloader:
            optimizer.zero_grad()

            outputs = vgg16_model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save the trained model checkpoint
    torch.save(vgg16_model.state_dict(), output_checkpoint_path)

if __name__ == "__main__":
    # Example usage:
    dataset_path = "../data"
    output_checkpoint_path = "../result/vgg16_checkpoint.pth"

    train_model(dataset_path, output_checkpoint_path)
