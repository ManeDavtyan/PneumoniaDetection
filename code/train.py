# # # Contains the training codes. It must accept a dataset path and hyperparameters as inputs.
# # # It should produce and save at least one checkpoint as output.


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SimpleVGG16
from dataset import PneumoniaDataset


def train_model(dataset_path, output_checkpoint_path, output_log_path, num_epochs=16, learning_rate=0.001,
                batch_size=32):
    # Set up your dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = PneumoniaDataset(dataset_path, transform, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the VGG16 model
    vgg16_model = SimpleVGG16()

    # Set up your loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(vgg16_model.parameters(), lr=learning_rate)

    # Lists to store loss and accuracy values
    losses = []
    accuracies = []

    # Train the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        vgg16_model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = vgg16_model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct_predictions += (predictions == labels.float().view_as(predictions)).sum().item()
            total_samples += labels.size(0)

            if (i + 1) % 10 == 0:
                print(f"Batch [{i + 1}/{len(train_dataloader)}], Loss: {loss.item()}")

        epoch_loss = running_loss / len(train_dataloader)
        accuracy = correct_predictions / total_samples

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss}, Accuracy: {accuracy}")

        # Save loss and accuracy values
        losses.append(epoch_loss)
        accuracies.append(accuracy)

    # Save the trained model checkpoint
    torch.save(vgg16_model.state_dict(), output_checkpoint_path)

    # Save loss and accuracy to a file
    with open(output_log_path, 'w') as f:
        for epoch, (loss, accuracy) in enumerate(zip(losses, accuracies)):
            f.write(f"Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}\n")


if __name__ == "__main__":
    # Example usage:
    dataset_path = "../data"
    output_checkpoint_path = "../result/simple_vgg16.pth"
    output_log_path = "../result/simple_vgg16_loss_accuracy.txt"

    train_model(dataset_path, output_checkpoint_path, output_log_path)


