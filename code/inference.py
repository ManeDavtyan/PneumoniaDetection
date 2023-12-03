#Contains the code inference code for processing a single image.
#It should take a single image path as input and save the output on Result folder.

# inference.py
import torch
from torchvision import transforms
from PIL import Image

def load_and_preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)

    return input_image

def predict_image(model, input_image):
    # Send the image through the model for prediction
    model.eval()
    with torch.no_grad():
        output = model(input_image)

    # Assuming binary classification (change if needed)
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

def inference(image_path, model):
    # Load and preprocess the image
    input_image = load_and_preprocess_image(image_path)

    # Get the prediction
    predicted_class = predict_image(model, input_image)

    return predicted_class
