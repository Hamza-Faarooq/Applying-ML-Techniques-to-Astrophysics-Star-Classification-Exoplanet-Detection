import pandas as pd
import numpy as np 

# Function to generate synthetic star data
def generate_star_data(num_samples=1000):
    np.random.seed(42)
    spectral_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    data = {
        'temperature': np.random.normal(5000, 2000, num_samples),
        'luminosity': np.random.normal(1, 0.5, num_samples),
        'radius': np.random.normal(1, 0.5, num_samples),
        'metallicity': np.random.normal(0, 0.1, num_samples),
        'spectral_type': np.random.choice(spectral_types, num_samples)
    }
    df = pd.DataFrame(data)
    return df

# Generate and save star data
star_data = generate_star_data()
star_data.to_csv('synthetic_star_data.csv', index=False)
print(star_data.head())