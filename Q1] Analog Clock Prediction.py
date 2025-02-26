#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import Required Libraries

# In[1]:


# Step 1: Import Required Libraries
# =============================================================================
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# # Step 2: Define Parameters and Paths

# In[2]:


# Step 2: Define Parameters and Paths
# =============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# Path to the folder containing analog clock images
IMAGE_FOLDER = "C://Users/saura/OneDrive/Desktop/MothersonProjectTemplate/Analog Clock/data/analog_clock_jpg/content/images/analog_clock_jpg"

# Path to the labels CSV file
CSV_PATH = "C://Users/saura/OneDrive/Desktop/MothersonProjectTemplate/Analog Clock/labels.csv"


# # Step 3: Read CSV and Prepare Data

# In[3]:


# Step 3: Read CSV and Prepare Data
# =============================================================================
# Load the CSV file with image filenames and time labels
df = pd.read_csv(CSV_PATH)
print("Dataset preview:")
print(df.head())

# Function to convert time string (HH:MM) to a normalized value [0,1]
def convert_time_to_normalized(time_str):
    time_str = str(time_str)
    hour, minute = map(int, time_str.split(':'))
    total_minutes = hour * 60 + minute
    return total_minutes / 1440.0

# Create a new column 'target' for normalized time
df['target'] = df['time'].apply(convert_time_to_normalized)


# # Step 4: Filter Out Unsupported Images

# In[4]:


# Step 4: Filter Out Unsupported Images
# =============================================================================
def check_image_validity(filename):
    full_path = os.path.join(IMAGE_FOLDER, filename)
    # Try alternative extension if file not found.
    if not os.path.exists(full_path):
        lower_path = full_path.lower()
        if lower_path.endswith('.jpg'):
            alt_path = full_path[:-4] + '.jpeg'
        elif lower_path.endswith('.jpeg'):
            alt_path = full_path[:-5] + '.jpg'
        else:
            return False
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            return False
    try:
        image_data = tf.io.read_file(full_path)
        _ = tf.image.decode_image(image_data, channels=3, expand_animations=False)
        return True
    except Exception:
        return False

df['is_valid'] = df['filename'].apply(check_image_validity)
df_valid = df[df['is_valid']].copy()
print(f"Total images: {len(df)}, Valid images: {len(df_valid)}")


# # Step 5: Split Data into Train, Validation, and Test Sets

# In[5]:


# Step 5: Split Data into Train, Validation, and Test Sets
# =============================================================================
# Adjust the splits based on the number of valid images
df_train = df_valid.iloc[:140].copy()
df_test  = df_valid.iloc[140:168].copy()   # 28 images
df_val   = df_valid.iloc[168:196].copy()      # 28 images

print("Train images:", len(df_train))
print("Test images:", len(df_test))
print("Validation images:", len(df_val))


# # Step 6: Create TensorFlow Data Pipeline

# In[6]:


# Step 6: Create TensorFlow Data Pipeline
# =============================================================================
def load_and_preprocess_image(filename, label):
    # Convert filename tensor to a Python string
    filename_str = filename.numpy().decode('utf-8')
    image_path = os.path.join(IMAGE_FOLDER, filename_str)
    # Check for alternative extension if needed
    if not os.path.exists(image_path):
        lower_path = image_path.lower()
        if lower_path.endswith('.jpg'):
            alt_path = image_path[:-4] + '.jpeg'
        elif lower_path.endswith('.jpeg'):
            alt_path = image_path[:-5] + '.jpg'
        else:
            alt_path = image_path
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(f"Neither {image_path} nor {alt_path} exists.")
    # Read and decode image data
    image_data = tf.io.read_file(image_path)
    try:
        image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    except Exception as e:
        raise ValueError(f"Error decoding image {image_path}: {e}")
    # Resize and preprocess image for MobileNetV2
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def load_and_preprocess_image_wrapper(filename, label):
    image, label = tf.py_function(
        func=load_and_preprocess_image,
        inp=[filename, label],
        Tout=[tf.float32, tf.float32]
    )
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    label.set_shape([])
    return image, label

def create_dataset(dataframe):
    filenames = dataframe['filename'].values
    labels = dataframe['target'].values.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_and_preprocess_image_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=100).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create TensorFlow datasets
train_ds = create_dataset(df_train)
val_ds = create_dataset(df_val)
test_ds  = create_dataset(df_test)


# # Step 7: Define Custom Metric (Time Accuracy)

# In[7]:


# Step 7: Define Custom Metric (Time Accuracy)
# =============================================================================
def time_accuracy(y_true, y_pred):
    # Prediction is "accurate" if within 5 minutes (normalized threshold)
    threshold = 5 / 1440.0
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.cast(diff < threshold, tf.float32))


# # Step 8: Build and Compile the Model

# In[8]:


# Step 8: Build and Compile the Model
# =============================================================================
# Load pre-trained MobileNetV2 as base model (without top classifier)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# Build the full model
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
# Output a single normalized value (using sigmoid activation)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

# Compile the model with MSE loss and custom time_accuracy metric
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',
              metrics=[tf.keras.metrics.MeanAbsoluteError(), time_accuracy])


# # Step 9: Train the Model

# In[9]:


# Step 9: Train the Model
# =============================================================================
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# # Step 10: Evaluate the Model on the Test Set

# In[10]:


# Step 10: Evaluate the Model on the Test Set
# =============================================================================
test_loss, test_mae, test_time_acc = model.evaluate(test_ds)
print("\nTest Performance:")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)
print("Test Time Accuracy (within 5 minutes):", test_time_acc)


# # Step 11: Generate Predictions and Convert to HH:MM Format

# In[11]:


# Step 11: Generate Predictions and Convert to HH:MM Format
# =============================================================================
predictions = model.predict(test_ds)

def normalized_to_time(norm_value):
    total_minutes = norm_value * 1440  # convert normalized value back to minutes
    hour = int(total_minutes // 60) % 24  # ensure hour is within 0-23
    minute = int(total_minutes % 60)
    return f"{hour:02d}:{minute:02d}"

# Retrieve filenames and true time values from the test split
test_filenames = df_test['filename'].values
test_true_times = df_test['time'].values

print("\nTest Predictions:")
pred_list = predictions.flatten()
for i, pred in enumerate(pred_list):
    pred_time = normalized_to_time(pred)
    true_time = test_true_times[i]
    print(f"Image: {test_filenames[i]}  |  True Time: {true_time}  |  Predicted Time: {pred_time}")


# # Step 12: Plot Training History

# In[12]:


# Step 12: Plot Training History
# =============================================================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'o-', label='Train Loss')
plt.plot(history.history['val_loss'], 'o-', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss over Epochs")
plt.legend()


# In[13]:


plt.subplot(1, 2, 2)
plt.plot(history.history['time_accuracy'], 'o-', label='Train Time Accuracy')
plt.plot(history.history['val_time_accuracy'], 'o-', label='Val Time Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (within 5 min)")
plt.title("Time Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




