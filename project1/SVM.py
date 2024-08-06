import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray


# Load the dataset
dataset = pd.read_csv("dataset.csv")

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100)) 
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array

# Preprocess images
X = np.array([preprocess_image(path) for path in dataset['Image_Path']])
y = dataset['Label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)

# Perform cross-validation
fold_number = 1
for train_index, val_index in stratified_kfold.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Flatten the images for SVM
    #X_train_flat = X_train.reshape(X_train.shape[0], -1)
    #X_val_flat = X_val.reshape(X_val.shape[0], -1)

    X_train_gray = np.array([rgb2gray(img) for img in X_train])
    X_val_gray = np.array([rgb2gray(img) for img in X_val])

    X_train_hog = np.array([hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in X_train_gray])
    X_val_hog = np.array([hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for img in X_val_gray])

    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_hog)
    X_val_pca = pca.transform(X_val_hog)

    # Define the SVM model
    svm_model = SVC(kernel='linear')r
    

    # Train the SVM model
    svm_model.fit(X_train_pca, y_train)

    # Make predictions on the validation set
    y_pred = svm_model.predict(X_val_pca)

    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Fold {fold_number} Accuracy: {accuracy}")

    # Evaluate additional metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, svm_model.decision_function(X_val_pca))
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"Fold {fold_number} Precision: {precision}")
    print(f"Fold {fold_number} Recall: {recall}")
    print(f"Fold {fold_number} F1 Score: {f1}")
    print(f"Fold {fold_number} AUROC: {auroc}")
    print(f"Fold {fold_number} Confusion Matrix:")
    print(conf_matrix)
    
    # Find the minimum and maximum values of the pac component to ensure the boundary 
    h = .02  # step size in the mesh
    x_min, x_max = X_val_pca[:, 0].min() - 1, X_val_pca[:, 0].max() + 1
    y_min, y_max = X_val_pca[:, 1].min() - 1, X_val_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot data points
    plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_val, cmap=plt.cm.coolwarm, edgecolors='k', marker='o')

    # Set plot labels
    plt.title(f"Fold {fold_number} - SVM Decision Boundary")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(f'image/SVM Fold{fold_number}.png')
    plt.show()

    # Continue with the rest of the evaluation metrics...

    fold_number += 1