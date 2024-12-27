# Mobile Application Traffic Classification

This project focuses on designing a network traffic representation and feature extraction pipeline for efficient mobile application identification. By leveraging multi-modality in network traffic captures, we can represent traffic features in a latent space to enable robust classification of mobile application traffic.

## Overview

Mobile application traffic classification is a critical task in modern networking, especially for mobile edge nodes that process traffic in real-time. This project employs the **UTM MobileNet2021 dataset**, preprocessing it to address challenges such as class imbalance and concept drifts. A balanced dataset is generated and used to train an **autoencoder model**, creating a latent space representation for traffic associated with different applications. Machine learning (ML) or deep learning (DL) classifiers are then used to classify the traffic into predefined application classes.

## Features

- **Multi-modal representation**: Utilizes data from multiple modalities in traffic captures.
- **Latent space generation**: Autoencoder-based feature extraction for dimensionality reduction.
- **Balanced dataset creation**: Preprocessing and data augmentation for robust model training.
- **Real-time traffic classification**: Designed for deployment in mobile edge nodes.

## Dataset

The ** dataset** is used for this project, containing network traffic captures for various applications. Preprocessing includes:

- Cleaning and filtering traffic data.
- Data augmentation to address concept drifts in network environments.
- Balancing the dataset for unbiased classification.

## Methodology

1. **Data Preprocessing**:
   - Clean traffic data to remove noise.
   - Generate balanced datasets through data augmentation.

2. **Feature Extraction**:
   - Train an autoencoder model on the preprocessed dataset.
   - Generate a latent space representation for traffic data.

3. **Classification**:
   - Use ML or DL classifiers (e.g., Random Forest, SVM, or CNN) to classify traffic into application classes.

## Model Architecture

### Autoencoder
The autoencoder processes traffic data to learn a compact latent space representation. Key features of the architecture include:
- Encoder: Extracts meaningful features.
- Decoder: Reconstructs traffic data for verification.

### Classifier
Trained on the latent space features, the classifier predicts the application class for a given traffic instance.

## Challenges

- **Concept drifts**: Addressed through robust preprocessing and data augmentation.
- **Multi-modality**: Utilized to improve classification performance.

## Results

The proposed pipeline successfully generates a multi-modal latent space and achieves high classification accuracy for mobile application traffic. The approach demonstrates scalability and robustness, making it suitable for real-time classification on mobile edge nodes.

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `pandas`, `tensorflow`, `scikit-learn`, `matplotlib`

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/mobile-traffic-classification.git
cd mobile-traffic-classification
pip install -r requirements.txt
