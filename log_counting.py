import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Set paths
test_dir = "C:/Users/daher/OneDrive/Desktop/University Materials/Semester 6/AI Engineering/Weel 5/test"
model_path = "C:/Users/daher/OneDrive/Desktop/University Materials/Semester 6/AI Engineering/Weel 5/code/resnet50_model.h5"

# Load the pre-trained model
model = load_model(model_path)

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),  # Adjust based on your model's input size
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)

# Initialize counters
total_logs = 0
log_counts = []

# Log counting logic
threshold = 0.4  # Adjusted threshold value
for i, prediction in enumerate(predictions):
    predicted_class = 'rust' if prediction[0] > threshold else 'no rust'
    log_count = 1 if predicted_class == 'rust' else 0  # Count as 1 for 'rust', 0 for 'no rust'

    log_counts.append(log_count)
    total_logs += log_count

    print(f"Image {i+1}: Predicted - {predicted_class}, Log Count - {log_count}")

print(f"Total logs counted: {total_logs}")

# Save results to CSV
results_df = pd.DataFrame({
    'Image': [f'Image {i+1}' for i in range(len(log_counts))],
    'Log Count': log_counts
})

results_df.to_csv('log_counts.csv', index=False)
print("Log counts saved to log_counts.csv.")
