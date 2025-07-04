import os
import requests
from PIL import Image
from transformers import pipeline

# Prevent OpenMP DLL conflict (Windows workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load zero-shot image classification pipeline with a compatible model
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# Load an image from a URL
url = "image.png"  # Replace with your image URL or local path
image = Image.open(requests.get(url, stream=True).raw)

# Candidate labels
candidate_labels = ["2 cats", "a plane", "a remote"]

# Run inference
outputs = classifier(image, candidate_labels)

# Print results
print(outputs)
# This code snippet is a simplified version of the original code that uses the Hugging Face Transformers library