import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.layers import Flatten
from keras.models import Model
import numpy as np
import csv
import ssl

# from weighted_pooling import weighted_history_pooling

ssl._create_default_https_context = ssl._create_unverified_context
# Load the VGG16 model pretrained on ImageNet
base_model = VGG16(weights="imagenet", include_top=False)

# Flatten the extracted features
flatten_layer = Flatten()(base_model.output)

# Create a new model with the Flatten layer
model = Model(inputs=base_model.input, outputs=flatten_layer)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract and flatten features for multiple images
def extract_and_flatten_features(image_paths):
    flattened_features_list = []
    for img_path in image_paths:
        img_array = preprocess_image(img_path)
        flattened_features = model.predict(img_array)
        flattened_features_list.append(flattened_features)
    return np.concatenate(flattened_features_list, axis=0)

# # All images paths to extract the visual features
# image_paths = [
#     "/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/images/_luvbebe_-1301263632084140443.jpg",
#     "/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/images/_luvbebe_-1309137756152393028.jpg",
#     "/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/images/_luvbebe_-1549849509901766222.jpg",
#     "/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/images/_luvbebe_-1598710622573229256.jpg",
#     "/Users/viru/Documents/GitHub/Micro-Influencer-Recommendations/influencer-dataset/images/_luvbebe_-1627337295786782351.jpg"
# ]

# # Extracting the visual features
# flattened_features = extract_and_flatten_features(image_paths)

# # Applying the Weighted History Pooling method to the visual features
# ev = weighted_history_pooling(flattened_features, 0.333)

# # Save visual features to CSV
# with open('embeddings/extracted_features_files/visual_features.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(ev)