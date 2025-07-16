import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# Step 1 & 2: Load and preprocess CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 3: Resize images to match MobileNetV2 input size (96x96)
X_train = tf.image.resize(X_train, [96, 96])
X_test = tf.image.resize(X_test, [96, 96])

# Step 4: Load pretrained MobileNetV2 without top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Step 5: Add custom classifier on top of base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Step 6: Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Step 7: Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# Step 9: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Step 10: Prediction and Visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict_sample(index):
    sample = tf.expand_dims(X_test[index], axis=0)  # Ensure shape (1, 96, 96, 3)
    prediction = model.predict(sample)
    pred_class = class_names[np.argmax(prediction)]
    actual_class = class_names[np.argmax(y_test[index])]
    plt.imshow(X_test[index].numpy())
    plt.title(f"Predicted: {pred_class} | Actual: {actual_class}")
    plt.axis('off')
    plt.show()

# Show a prediction example
predict_sample(7)

# Step 11: Save the trained model
model.save("transfer_learning_cifar10_model.h5")
