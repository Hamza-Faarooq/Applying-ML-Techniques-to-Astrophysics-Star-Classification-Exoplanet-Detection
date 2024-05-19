import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the star classification data
star_data = pd.read_csv('synthetic_star_data.csv')

# Summary statistics
print(star_data.describe())

# Pairplot for visualizing the data
sns.pairplot(star_data, hue='spectral_type')
plt.show()