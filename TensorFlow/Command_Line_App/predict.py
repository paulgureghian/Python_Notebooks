# Created by Paul A. Gureghian in Mar 2022.

# Imports 
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Define variables 
batch_size = 32 
image_size = 224

class_names = {}

# Read in the label_map file.
json_file = open('label_map.json')
label_map = json.load(json_file)
#print(label_map) 

# Resize and normalize the input.
def process_image(image):
    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    
    return image

# Make predictions.
def predict(test_images, model, top_k):
    
    test_images = np.array(test_images)
    processed_test_images = process_image(test_images)
    expanded_processed_test_images = np.expand_dims(processed_test_images, axis=0)
    predictions = keras_model.predict(expanded_processed_test_images)
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    
    print("The 'probs' are: ", top_k_values)
    print("The 'classes' are: ", top_k_indices)
    
    return top_k_values, top_k_indices

# More imports.
import glob
import matplotlib.image as mpimg

# Call the predict function and print the results.
for image in glob.glob('/test_images/*'):

 print(image)   
 test_images = Image.open(image)
    
 top_k_values, top_k_indices = predict(test_images, keras_model, 5)
 
 flower_names = []
 print("Class names:")
    
 for idx in top_k_indices[0]:
    
    print("-", label_map[str(idx + 1)])
    flower_names.append(label_map[str(idx + 1)])
    
# Implement the 'argparse' module.
if __name__ == '__main__':
    
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')
    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    test_images = args.arg1
    keras_model = args.arg2
    top_k = args.top_k
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        
    probs, classes = predict(test_images, keras_model, top_k)
    
    print("The probabilities are: ", probs)
    print("The classes are: ", classes)
   

















