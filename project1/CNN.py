import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv("dataset.csv")

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100))  
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array

# Preprocess images and convert labels to one-hot encoding
X = np.array([preprocess_image(path) for path in dataset['Image_Path']])
y = to_categorical(dataset['Label'])

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)

# Define the CNN model
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Perform cross-validation
fold_number = 1
for train_index, val_index in stratified_kfold.split(X, np.argmax(y, axis=1)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]


    # Train the model
    model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    # Evaluate the model on the validation set
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    
    # Plot points with different colors for original labels
    plt.scatter(np.arange(len(y_val)), y_pred_prob[:, 1], c=y_val[:, 1], cmap='viridis', alpha=0.5)
    plt.xlabel('Data Points')
    plt.ylabel('Probability for Label 1')
    plt.title(f'Fold {fold_number} - Original Labels vs Predicted Probabilities')
    plt.colorbar()

    plt.savefig(f'image/CNN-Fold {fold_number}.png')
    plt.show()


    # Calculate evaluation metrics
    accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred)
    precision = precision_score(np.argmax(y_val, axis=1), y_pred)
    recall = recall_score(np.argmax(y_val, axis=1), y_pred)
    f1 = f1_score(np.argmax(y_val, axis=1), y_pred)
    auroc = roc_auc_score(y_val, y_pred_prob,multi_class='ovr')
    conf_matrix = confusion_matrix(np.argmax(y_val, axis=1), y_pred)

    print(f"Fold {fold_number} Metrics:")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1 Score: {f1}")
    print(f"  AUROC: {auroc}")
    print(f"Fold {fold_number} Confusion Matrix:")
    print(conf_matrix)


    fold_number += 1
