##
Applying-ML-Techniques-to-Astrophysics-Star-Classification-Exoplanet-Detection

# Exoplanet Detection with Machine Learning

## Overview
- This project aims to detect exoplanets using machine learning techniques applied to synthetic light curve data.
- Exoplanets are planets outside our solar system, and their detection is crucial for understanding the prevalence and diversity of planetary systems in the universe.
- The project involves generating synthetic light curve data, which simulates the brightness variations of stars over time, including transit events caused by orbiting exoplanets.
- A machine learning model, specifically an LSTM neural network, is trained on this data to identify patterns indicative of exoplanetary transits.



## Data
- The synthetic light curve data used in this project is generated programmatically using the generate_data.py script. 
- It includes time-series data representing the brightness variations of stars, with added transit events to simulate exoplanet transits.
- **Features**: The features used in the project are the brightness values of stars over time, represented as time-series data. No additional feature engineering is performed, as the raw light curve data contains relevant information for exoplanet detection.


## Models
- The machine learning model implemented for exoplanet detection is a Long Short-Term Memory (LSTM) neural network.
- LSTM networks are well-suited for sequential data like time-series, making them suitable for analyzing light curve data.
- The model architecture consists of one LSTM layer followed by a dense layer with a sigmoid activation function for binary classification.


## Results
- The trained LSTM model achieves promising results on the test set, as indicated by evaluation metrics such as precision, recall, and F1 score. 
- These metrics provide insights into the model's ability to correctly identify exoplanetary transits from synthetic light curve data.
- Additionally, training history plots, showing the model's performance over epochs, are provided in the project directory for further analysis.
