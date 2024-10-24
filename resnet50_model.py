import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Set directories
base_dir = r'C:\Users\daher\OneDrive\Desktop\University Materials\Semester 6\AI Engineering\Weel 5'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Check GPU availability
print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")

# Image parameters
img_height, img_width = 150, 150
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='binary',
    shuffle=False  # Keep the order consistent
)

# Build the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers (optional: useful for transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
)

# Evaluate the model
print("Evaluating ResNet50 model on the test set...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"ResNet50 Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model_path = os.path.join(base_dir, 'code', 'resnet50_model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")

# Generate predictions and save them to a CSV file
csv_path = os.path.join(base_dir, 'resnet50_test', 'resnet50_predictions.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Predicted Class', 'Actual Class'])

    for i in range(len(test_generator)):
        img, label = test_generator[i]
        prediction = model.predict(img)[0][0]
        predicted_class = 'rust' if prediction > 0.5 else 'no rust'
        actual_class = 'rust' if label[0] == 1 else 'no rust'
        writer.writerow([f'Image {i + 1}', predicted_class, actual_class])

print(f"Predictions saved to {csv_path}")
