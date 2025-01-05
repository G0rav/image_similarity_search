# Importing torch
# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image


model_path = "clip_finetuend_model.pth"  # Replace with the path to finetuned model
query_image_path = '/content/1.png'  # Replace with the path to your query image

# setting device
# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading and transforming the Fashion MNIST dataset for CLIP compatibility
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for CLIP
    transforms.Resize((224, 224)),  # Resize to match CLIP input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)


# Select a subset of 5000 images for training
train_subset_size = 5000
indices = torch.randperm(len(train_dataset))[:train_subset_size]  # Randomly select 5000 indices
train_subset = Subset(train_dataset, indices)


# Select a subset of 5000 images for training
test_subset_size = 1000
indices1 = torch.randperm(len(test_dataset))[:test_subset_size]  # Randomly select 5000 indices
test_subset = Subset(test_dataset, indices1)


# DataLoader setup for batching the data for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Define the same model architecture as in the training script
class CLIPFashionMNIST(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPFashionMNIST, self).__init__()
        self.clip = clip_model.vision_model  # Use CLIP's vision model as a feature extractor
        self.fc1 = nn.Linear(clip_model.config.vision_config.hidden_size, 64)  # First hidden layer
        self.batch_norm1 = nn.BatchNorm1d(64)  # Batch Normalization
        self.dropout = nn.Dropout(0.25)  # Dropout with 50% probability
        self.fc2 = nn.Linear(64, num_classes)  # Output layer

    def forward(self, images):
        vision_outputs = self.clip(pixel_values=images)  # Extract features using CLIP
        x = self.fc1(vision_outputs.pooler_output)  # Apply first linear layer
        x = torch.relu(x)  # Apply ReLU activation
        x = self.batch_norm1(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        logits = self.fc2(x)  # Output logits
        return logits

# Load the pre-trained CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Initialize the model
num_classes = 10  # Fashion MNIST has 10 classes
model = CLIPFashionMNIST(clip_model, num_classes).to(device)

# Load the saved state dictionary
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()


def extract_embeddings_from_batches(model, dataloader, device, num_batches=5):
    """
    Extract embeddings and labels from a limited number of batches.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        device (torch.device): Device to run the model on.
        num_batches (int): Number of batches to process.

    Returns:
        tuple: (embeddings, labels) where embeddings is a torch.Tensor of shape
               (num_samples, embedding_dim) and labels is a torch.Tensor of shape (num_samples,)
    """
    model.eval()  # Set the model to evaluation mode
    all_embeddings = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting Embeddings")):
            if i >= num_batches:
                break  # Stop after processing the specified number of batches

            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Get embeddings from the model
            vision_outputs = model.clip(pixel_values=images)
            embeddings = vision_outputs.pooler_output

            # Collect embeddings and labels
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_images.append(images.cpu())

    # Stack all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_images = torch.cat(all_images, dim=0)

    return all_embeddings, all_labels, all_images

# Extracting embeddings
all_embeddings, all_labels, all_images = extract_embeddings_from_batches(model, test_loader, device, num_batches=5)

print(f"Extracted Embeddings Shape: {all_embeddings.shape}")
print(f"Extracted Labels Shape: {all_labels.shape}")
print(f"Extracted Images Shape: {all_images.shape}")


def load_image(image_path):
    """
    Load and preprocess the image.
    Args:
        image_path (str): Path to the image.
    Returns:
        torch.Tensor: Processed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the model's input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Add batch dimension

def get_image_embedding(model, image, device):
    """
    Get the embedding for a single image using the trained model.
    Args:
        model (torch.nn.Module): The trained model.
        image (torch.Tensor): The image tensor.
        device (torch.device): The device to run the model on.
    Returns:
        torch.Tensor: The image embedding.
    """
    model.eval()  # Set the model to evaluation mode
    image = image.to(device)  # Move image to device

    with torch.no_grad():
        vision_outputs = model.clip(pixel_values=image)
        embedding = vision_outputs.pooler_output  # Get the embedding (pooled output)

    return embedding.cpu()

def find_similar_images_for_query(query_image, all_embeddings, all_images, top_k=5):
    """
    Find and display the most similar images to a query image based on CLIP embeddings.
    Args:
        query_image (torch.Tensor): The query image tensor.
        all_embeddings (torch.Tensor): Normalized embeddings of all images.
        all_images (torch.Tensor): Tensor of all images (C, H, W format).
        top_k (int): Number of similar images to display.
    Returns:
        None
    """
    # Normalize embeddings
    normalized_embeddings = F.normalize(all_embeddings, p=2, dim=1)

    # Get embedding for the query image
    query_embedding = get_image_embedding(model, query_image, device)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # Compute cosine similarities between the query image and all other embeddings
    similarities = (normalized_embeddings @ query_embedding.T).squeeze(1).cpu().numpy()

    # Get indices of the top-k most similar images
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Plot the query image and the most similar images
    plt.figure(figsize=(15, 3))
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.title("Query Image")
    plt.axis("off")

    for i, idx in enumerate(top_indices):
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(all_images[idx].permute(1, 2, 0).cpu().numpy())
        plt.title(f"Similar {i+1}")
        plt.axis("off")

    plt.show()

# Example Usage:
def test_script(query_image_path, model, all_embeddings, all_images, device, top_k=5):
    """
    Test script to accept a query image path and display similar images.
    Args:
        query_image_path (str): Path to the query image.
        model (torch.nn.Module): The trained model.
        all_embeddings (torch.Tensor): All image embeddings.
        all_images (torch.Tensor): All images.
        device (torch.device): The device to run the model on.
        top_k (int): Number of similar images to retrieve.
    """
    query_image = load_image(query_image_path)  # Load and preprocess the image
    find_similar_images_for_query(query_image, all_embeddings, all_images, top_k)


# Example Usage (replace with actual paths and model):
test_script(query_image_path, model, all_embeddings, all_images, device, top_k=5)
