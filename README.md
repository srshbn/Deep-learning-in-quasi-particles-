# Deep-learning-in-quasi-particles
Predicting Polaritonic Wavelengths from Interference Patterns
This project demonstrates how to generate synthetic polaritonic interference patterns and train a Convolutional Neural Network (CNN) to estimate the wavelengths of two overlapping wave modes.

# Description
Simulates images of superposed sine waves with random wavelengths and noise.
Generates a dataset of labeled examples for training.
Defines and trains a CNN using TensorFlow/Keras to regress the two wavelengths from each image.
Tests the model on new synthetic data and compares predicted vs. actual values.
Visualizes results with example images and regression fits.
# Output
Predicted vs. actual wavelength pairs on sample test images.
Scatter plots comparing ground truth and predicted values.
Nonlinear and linear fits to assess model performance (with R² and MSE).
# Requirements
Install the required Python packages:
pip install numpy matplotlib tensorflow scipy scikit-learn
# Steps: 
Generate the dataset
Train the CNN
Evaluate the model
Display visual output with predictions and fit analysis
# Customize
Adjust the range of wavelengths to simulate different modal behaviors.
Modify network architecture or training parameters to tune performance.
Add noise or smoothing to simulate real experimental conditions.
This script is ideal for prototyping ML-based analysis of wave interference patterns or learning about CNN regression tasks.
