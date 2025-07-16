import tensorflow as tf
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Use a smaller subset of training data to speed up training
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

# Define a smaller and faster model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Measure training time
start_time = time.time()
model.fit(X_train_small, y_train_small, epochs=3, batch_size=128, validation_data=(X_test, y_test))
end_time = time.time()

print(f"\nTraining time: {end_time - start_time:.2f} seconds")
