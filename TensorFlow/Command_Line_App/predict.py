# Created by Paul A. Gureghian in Mar 2022.

# Imports 
import os
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


# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
def predict(test_images, keras_model, top_k):
    
    test_images = Image.open(test_images)
    print(type(test_images))
    test_images = np.array(test_images)
    processed_test_images = process_image(test_images)
    expanded_processed_test_images = np.expand_dims(processed_test_images, axis=0)
    print("Test images shape: ", expanded_processed_test_images.shape)
    predictions = keras_model.predict(expanded_processed_test_images)
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    
    print("The 'probs' are: ", top_k_values)
    print("The 'classes' are: ", top_k_indices)
    
    return top_k_values, top_k_indices
   
    
# Implement the 'argparse' module.
# Call the predict function.
if __name__ == '__main__':
    
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--category_names')
    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    test_images = args.arg1
    keras_model = args.arg2
    keras_model = tf.keras.models.load_model(keras_model, compile=False, custom_objects={'KerasLayer':hub.KerasLayer})
    top_k = args.top_k
           
    probs, classes = predict(test_images, keras_model, top_k)
    
    print("The probabilities are: ", probs)
    print("The classes are: ", classes)
   
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        
        flower_names = []
        print("Class names:")
        
        for idx in classes[0]:
            print("-", class_names[str(idx + 1)])
            
        
        
        
        
        


