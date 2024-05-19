import matplotlib.pyplot as plt
import pandas as pd

# Load the exoplanet detection data
light_curve_data = pd.read_csv('synthetic_light_curve_data.csv')

# Example of light curve visualization
plt.figure(figsize=(10, 6))
plt.plot(eval(light_curve_data['light_curve'][0]))
plt.title('Example Light Curve')
plt.xlabel('Time')
plt.ylabel('Relative Brightness')
plt.show()