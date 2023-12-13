
# import torch
# from torchvision import transforms, datasets
# import os
#
# # Define the data directory
# data_dir = '../data'
#
# # Define the transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # Load the datasets
# train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
# test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
# val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
#
# # Create data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#
# # Example usage:
# # Iterate over the train loader
# for inputs, labels in train_loader:
#     # Your training code here
#     print(f"Batch shape: {inputs.shape}, Labels: {labels}")
#     break  # Remove this break statement if you want to iterate over the entire dataset










# #example usage of model
# from code.model import VGG16
#
#
# # Instantiate the model
# vgg16_model = VGG16()
#
# # Print the model architecture
# print(vgg16_model)
