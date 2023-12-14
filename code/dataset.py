# # # Includes all code related to the dataset,
# # # such as the dataset class, preprocessing,
# # # augmentation, and post-processing routines.


import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.image_paths, self.labels = self.load_dataset(train)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label

    def load_dataset(self, train):
        image_paths = []
        labels = []

        split_dir = 'train' if train else 'test'  # Adjust as needed
        classes = os.listdir(os.path.join(self.data_dir, split_dir))

        for class_name in classes:
            class_dir = os.path.join(self.data_dir, split_dir, class_name)
            if os.path.isdir(class_dir):
                class_label = 1 if class_name == "PNEUMONIA" else 0

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):  # Check if it's a file
                        image_paths.append(img_path)
                        labels.append(class_label)

        return image_paths, labels

    # Add functions for preprocessing, augmentation, and post-processing
    def preprocess(self, img):
        # Add preprocessing steps if needed
        return img

    def augment(self, img):
        # Add data augmentation steps if needed
        return img

    def postprocess(self, img):
        # Add post-processing steps if needed
        return img

# Example usage:
# data_dir = "../data"
# transform = transforms.Compose([transforms.Resize((224, 224)),
#                                 transforms.ToTensor()])
# pneumonia_dataset = PneumoniaDataset(data_dir, transform)
# image, label = pneumonia_dataset[0]
