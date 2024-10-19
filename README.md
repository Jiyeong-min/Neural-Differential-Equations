# AdvancedPathModel: Learning Path Representation with UMAP

## Project Overview

This project implements the code related to **Figure 4: Examples of the original path (X) and learnt path (Y) using UMAP**. 
The goal is to visualize the learned path in comparison with the original sine wave path using UMAP dimensionality reduction.

## Key Features

- **Data Generation**: Generates a noisy sine wave as the target path for the model to learn.
- **Model Architecture**: A fully connected neural network with 5 layers.
- **Training Process**: The model is trained using MSE loss and an Adam optimizer with weight decay.

## Model Architecture

The `AdvancedPathModel` consists of the following layers:

- **Fully Connected Layers**: 5 layers of increasing and decreasing units (256, 128, 64, 32, 1).
- **Dropout**: Applied after the first layer to prevent overfitting.
