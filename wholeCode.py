import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# Step 1: Generate synthetic light curve data
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

# Step 2: Load and preprocess the data
light_curve_data = pd.read_csv('synthetic_light_curve_data.csv')

# Convert light curve data from string to list
X = np.array(light_curve_data['light_curve'].apply(eval).tolist())
y = light_curve_data['label'].values

# Reshape data for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()