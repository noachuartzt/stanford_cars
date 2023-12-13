"""This file contains the auxiliary functions to plot images and evaluation metrics."""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import settings

def plot_evaluation(history: dict) -> None:
    """Plot the accuracy and loss of the model."""
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy'), plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])   
    plt.title('Model Loss')
    plt.ylabel('Loss'), plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_samples(df: pd.DataFrame, ncols: int = 3, nrows: int = 2) -> None:
    """Plot sample images from the dataset."""
    
    # Visualize sample images
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 10))

    for ax in axs.flatten():
        
        # Choose random number
        random_index = np.random.randint(0, len(df))

        # Get car image path
        car_path = df.iloc[random_index, 0]

        # Get car model name
        car_model = df.iloc[random_index, -1]
        
        path = settings.train + df['model'][random_index] + '/' + car_path
        ax.imshow(plt.imread(path))
        ax.set_title(f"{car_model}")
            
        ax.axis('off')
        
        
def plot_imgs_from_generator(train_generators: dict, ncols: int = 3, nrows: int = 1) -> None:
    """Plot sample images from the generators."""
    
    # Put the code above in a subplot
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))

    for i, (img_size, img) in enumerate(train_generators.items()):
        # Get a random image from the generator
        img, label = random.choice(img)
        class_labels = {v: k for k, v in img.class_indices.items()} 

        # Plot the image
        ax[i].imshow(img[i])
        ax[i].set_title(f'{img_size}x{img_size} - {class_labels[np.argmax(label[i])]}')
        
        
def plot_comparison(df_test, model_names, predicted_classes):
    """Plot a comparison between the actual and predicted images."""
    
    figs, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

    for i in range(3):  # Iterate over columns
        # Choose random number
        random_index = np.random.randint(0, len(df_test))

        # Get car image path
        car_path = df_test.iloc[random_index, 0]

        # Get car model name
        car_model = df_test.iloc[random_index, -1]

        # Predict car model
        prediction = model_names[predicted_classes[random_index]]

        # Paths
        path = settings.test + df_test['model'][random_index] + '/' + car_path

        pred_path = settings.train + prediction
        pred_path = pred_path + '/' + os.listdir(pred_path)[0]

        # REAL IMAGE
        axs[0, i].imshow(plt.imread(path))
        axs[0, i].set_title(f"ACTUAL: {car_model}")
        axs[0, i].axis('off')

        # PREDICTED IMAGE
        axs[1, i].imshow(plt.imread(pred_path))
        axs[1, i].set_title(f"PREDICTION: {prediction}")
        axs[1, i].axis('off')

    plt.show()