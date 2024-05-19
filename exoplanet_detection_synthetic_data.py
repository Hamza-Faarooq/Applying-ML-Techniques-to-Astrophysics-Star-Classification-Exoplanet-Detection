import numpy as np
import pandas as pd

# Function to generate synthetic light curve data
def generate_light_curve(num_samples=1000, length=200):
    np.random.seed(42)
    data = []
    for _ in range(num_samples):
        light_curve = np.random.normal(1, 0.01, length)  # Baseline light curve with small noise
        # Randomly add a transit event
        if np.random.rand() > 0.5:
            transit_start = np.random.randint(50, 150)
            transit_depth = np.random.uniform(0.98, 0.99)
            light_curve[transit_start:transit_start+10] *= transit_depth
            label = 1
        else:
            label = 0
        data.append((light_curve.tolist(), label))
    
    df = pd.DataFrame(data, columns=['light_curve', 'label'])
    return df

# Generate and save light curve data
light_curve_data = generate_light_curve()
light_curve_data.to_csv('synthetic_light_curve_data.csv', index=False)
print(light_curve_data.head())