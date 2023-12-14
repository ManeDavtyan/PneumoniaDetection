# # # Contains the code inference code for processing a single image.
# # # It should take a single image path as input and save the output on Result folder.



import torch
from torchvision import transforms
from PIL import Image
from code.model import SimpleVGG16

def inference_single_image(image_path, model, save_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Convert the output to a probability (assuming binary classification)
    probability = torch.sigmoid(output).item()

    # Classify based on a threshold (you can adjust this threshold as needed)
    predicted_class = 1 if probability > 0.5 else 0

    # Save the result
    result_text = f"Predicted Class: {predicted_class}, Probability: {probability:.4f}"
    with open(save_path, 'w') as result_file:
        result_file.write(result_text)

if __name__ == "__main__":
    # Example usage:
    image_path = '../data/test/NORMAL/IM-0075-0001.jpeg'

    save_path = "../result/IM-0075-0001_result.txt"

    # Instantiate the VGG16 model
    vgg16_model = SimpleVGG16()

    # Load the trained weights 
    vgg16_model.load_state_dict(torch.load('../result/simple_vgg16.pth'))

    # Run inference and save the result
    inference_single_image(image_path, vgg16_model, save_path)

