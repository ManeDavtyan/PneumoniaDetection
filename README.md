# Pneumonia Detection : SimpleVGG16

## Objective
The primary objective of this project is to implement and train a deep learning model for the detection of pneumonia from chest X-ray images. The model, a simplified version of the VGG16 architecture named `SimpleVGG16`, is trained on a provided dataset containing images of individuals with and without pneumonia.


## Project Implementation
The main goal of this project is to create a Pneumonia detection model using a simplified VGG16 architecture implemented from scratch. The model is designed to classify chest X-ray images of patients as either having pneumonia or not. The project encompasses the following key components:

1. **Dataset Handling (`code/dataset.py`):**
   - The `dataset.py` file includes code related to the dataset, such as the dataset class, preprocessing, augmentation, and post-processing routines.

2. **Model Architecture (`code/model.py`):**
   - The `model.py` file contains code defining the model architecture. In this case, we are implementing a simplified VGG16 model (`SimpleVGG16`) from scratch.

3. **Training Script (`code/train.py`):**
   - The `train.py` script is responsible for training the `SimpleVGG16` model. It accepts the dataset path, hyperparameters, and outputs at least one checkpoint as a result.

4. **Inference Script (`code/inference.py`):**
   - The `inference.py` script processes a single image for inference using the trained `SimpleVGG16` model. It takes an image path, loads the model checkpoint, and saves the output in the `results/` folder.

5. **Usage of SimpleVGG16:**
   - The `SimpleVGG16` model is a simplified version of the VGG16 architecture, with custom modifications. It is implemented from scratch and trained on the provided dataset.
   
   ## SimpleVGG16 Architecture

| Layer             | Operation                                  |
|-------------------|--------------------------------------------|
| Input             | 3x244x244 (RGB image)                      |
| **Convolution 3x3**| 64 filters, ReLU activation                |
| **MaxPool 2x2**    |                                            |
| **Convolution 3x3**| 128 filters, ReLU activation               |
| **MaxPool 2x2**    |                                            |
| **Convolution 3x3**| 256 filters, ReLU activation               |
| **MaxPool 2x2**    |                                            |
| **Convolution 3x3**| 512 filters, ReLU activation               |
| **MaxPool 2x2**    |                                            |
| **Convolution 3x3**| 512 filters, ReLU activation               |
| **MaxPool 2x2**    |                                            |
| **Fully Connected**| 4096 neurons, ReLU activation              |
| **Fully Connected**| 4096 neurons, ReLU activation              |
| **Fully Connected**| 1 neuron, for binary classification        |
| **Output**        | 1x1 (Binary classification output)          |


6. **Result Storage (`results/`):**
   - The `results/` folder is used for storing model checkpoints, logs, and inference results.

The project provides a comprehensive workflow for developing, training, and using a Pneumonia detection model based on the simplified VGG16 architecture.


## Requirements

Make sure you have the following dependencies installed before running the code.
```{python}
Pillow==10.1.0
torch==2.1.1
torchvision==0.16.1
```

To have them, you can simply install requirnments file. 
```{python}
pip install -r requirements.txt
```


## Usage

1. **Training the Model:**
   - Train the `SimpleVGG16` model on your dataset using the provided script. Adjust the paths and hyperparameters as needed. Initially, 16 epochs will be run, while saving the loss and accuracy values in a .txt file in result folder. 

```{python}
python code/train.py --dataset_path ../data --output_checkpoint_path ../results/model_checkpoint.pth
```

2. **Inference on Single Image:**

Run the inference script to classify a single image. Replace /path/to/your/image.jpg with the actual image path.
```{python}
python code/inference.py --image_path /path/to/your/image.jpg --model_path ../results/model_checkpoint.pth --output_path ../results/inference_result.txt

```

3. **Usage of SimpleVGG16:**

The SimpleVGG16 model is a simplified version of the VGG16 architecture, with custom modifications. It is implemented from scratch and trained on the provided dataset.


```{python}
from code.model import SimpleVGG16
import torch

# Instantiate the model
model = SimpleVGG16()

# Load the trained weights
model.load_state_dict(torch.load('../results/model_checkpoint.pth'))

# Perform inference on an example image
# (Ensure the image is preprocessed as required)
output = model(example_input_tensor)

# Interpret the output as needed

```

### Conclusion

This project aims to contribute to the early diagnosis of pneumonia through the application of deep learning techniques. The provided model and scripts offer a foundation for further research and development in medical image analysis.

You can find the chest X-ray images dataset here -> https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
