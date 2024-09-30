import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage.feature import hog
from skimage import exposure

def extract_features(image_path, resize_to=(100, 100)):
    """
    Extract features from an image.

    Parameters:
    - image_path: Path to the image file.
    - resize_to: New size after resizing, represented as a tuple (width, height).

    Returns:
    - 1D numpy array representing the features.
    """
    try:
        with Image.open(image_path) as img:
            img = img.resize(resize_to)
            
            img_array = np.array(img).flatten()

       
        img = cv2.imread(image_path)
        img = cv2.resize(img, resize_to)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        fd, hog_img = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
        hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        hog_features = fd.flatten()

        combined_features = np.concatenate((img_array, hog_features))

        return hog_features
    except Exception as e:
        print(f"Error occurred while extracting features: {e}")
        return None

def cluster_images(csv_path, num_clusters=2):
    """
    Cluster images.

    Parameters:
    - csv_path: Path to the CSV file containing image paths and labels.
    - num_clusters: Number of clusters.

    Returns:
    - A dictionary containing the path of each image and its corresponding cluster label.
    """
    # Read the CSV file    
    dataset = pd.read_csv(csv_path)

    features = []
    image_paths = []
    labels = []

    # Iterate over all images in the dataset
    for index, row in dataset.iterrows():
        image_path = row['Image_Path']
        label = row['Label'] 
        

        # Check if the file is an image
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature = extract_features(image_path)

            # Ensure successful feature extraction
            if feature is not None:
                features.append(feature)
                image_paths.append(image_path)
                labels.append(label)

    # Convert features to a NumPy array
    features_array = np.array(features)


    # Perform feature transformation using PCA
    num_pca_components=2
    pca = PCA(n_components=num_pca_components)
    pca_result = pca.fit_transform(features_array)

    kmeans = KMeans(n_clusters=num_clusters, random_state=21)  # Remove 'random_state' for randomness
    kmeans.fit(pca_result)
    centers = kmeans.cluster_centers_

    clusters = kmeans.predict(pca_result)

    # Associate each image with its cluster label, label, and PCA components
    image_cluster_label_mapping = {'Image_Path': image_paths, 'Cluster': clusters, 'Label': labels, 
            'PCA_Component_1': pca_result[:, 0], 'PCA_Component_2': pca_result[:, 1]}

    return image_cluster_label_mapping,centers


def visualize_clusters(image_cluster_mapping,centers):
    """
    Visualize clustering results.

    Parameters:
    - image_cluster_mapping: A dictionary containing the path of each image and its corresponding cluster label.
    """
    # Convert the dictionary to separate lists
    image_paths = image_cluster_mapping['Image_Path']
    clusters = image_cluster_mapping['Cluster']
    labels=image_cluster_mapping['Label']

    # Visualize the clustering results
    """plt.figure(figsize=(10, 6))
    plt.scatter(range(len(image_paths)), clusters, c=labels, cmap='viridis')
    plt.xlabel('Image Index')
    plt.ylabel('Cluster Label')
    plt.title('Image Clustering')
    plt.savefig('image/PCA-image_index.png')
    plt.show()"""

    df = pd.DataFrame(image_cluster_mapping)

    # Calculate the frequency of different labels assigned to different clusters
    count_df = df.groupby(['Label', 'Cluster']).size().reset_index(name='Count')

    print("Label Clustering Statistics:")
    print(count_df)

    
    
    # Plot the chart of PCA components
    pca_component_1 = image_cluster_mapping['PCA_Component_1']
    pca_component_2 = image_cluster_mapping['PCA_Component_2']

    # Create colors for different clusters
    unique_clusters = set(clusters)
    colors = [plt.cm.jet(cluster / float(len(unique_clusters) - 1)) for cluster in clusters]

    # Plot the scatter plot
    plt.scatter(pca_component_1, pca_component_2, c=colors, marker='o', alpha=0.5)    
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Components Scatter Plot')
    plt.savefig('image/PCA-scatter.png')
    plt.show()


csv_file_path = "dataset.csv"  
num_clusters = 2  


# Perform clustering
image_cluster_mapping,centers = cluster_images(csv_file_path, num_clusters)

# Visualize clustering results
visualize_clusters(image_cluster_mapping,centers)
