import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import pandas as pd

def extract_features(image_path, resize_to=(100, 100)):
    """
    從圖片中提取特徵

    Parameters:
    - image_path: 圖片文件路徑
    - resize_to: 調整大小後的新大小，以元組 (width, height) 表示

    Returns:
    - 一維numpy數組表示的特徵
    """
    try:
        # 打開並調整大小
        with Image.open(image_path) as img:
            img = img.resize(resize_to)

            # 轉換為灰度圖
            img = img.convert('L')

            # 將圖片轉換為一維numpy數組
            img_array = np.array(img).flatten()

            return img_array
    except Exception as e:
        print(f"提取特徵時發生錯誤：{e}")
        return None

def cluster_images(csv_path, num_clusters=3):
    """
    對圖片進行聚類

    Parameters:
    - csv_path: 包含圖片路徑和標籤的CSV文件路徑
    - num_clusters: 聚類數量

    Returns:
    - 一個字典，包含每個圖片的路徑和其所屬的聚類標籤
    """
    # 讀取CSV文件
    dataset = pd.read_csv(csv_path)

    features = []
    image_paths = []
    labels = []

    # 遍歷資料中的所有圖片
    for index, row in dataset.iterrows():
        image_path = row['Image_Path']
        label = row['Label']  # 假設標籤欄位名稱為 'Label'

        # 檢查檔案是否為圖片
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature = extract_features(image_path)

            # 確保成功提取特徵
            if feature is not None:
                features.append(feature)
                image_paths.append(image_path)
                labels.append(label)

    # 將特徵轉換為NumPy數組
    features_array = np.array(features)

    selector = SelectKBest(f_classif, k=2)
    selected_features = selector.fit_transform(features_array, labels)


    kmeans = KMeans(n_clusters=num_clusters, random_state=22)
    clusters = kmeans.fit_predict(features_array)

    # 將每個圖片與其所屬的聚類標籤和標籤對應起來
    image_cluster_label_mapping = {'Image_Path': image_paths, 'Cluster': clusters, 'Label': labels}
    #image_cluster_label_mapping = dict(zip(image_paths, clusters))

    return image_cluster_label_mapping

def visualize_clusters(image_cluster_mapping):
    """
    將聚類結果視覺化

    Parameters:
    - image_cluster_mapping: 一個字典，包含每個圖片的路徑和其所屬的聚類標籤
    """
    # 將字典轉換為兩個分離的列表
    #image_paths, clusters = zip(*image_cluster_mapping.items())
    image_paths = image_cluster_mapping['Image_Path']
    clusters = image_cluster_mapping['Cluster']
  
    print(len(image_paths))
    print(len(clusters))

    # 將聚類結果視覺化
    plt.figure(figsize=(10, 6))
    clusters = list(clusters)
    plt.scatter(range(len(image_paths)), clusters, c=clusters, cmap='viridis')
    plt.xlabel('Image Index')
    plt.ylabel('Cluster Label')
    plt.title('Image Clustering')
    plt.show()

    df = pd.DataFrame(image_cluster_mapping)

    # 統計相同標籤被分配到不同聚類的次數
    count_df = df.groupby(['Label', 'Cluster']).size().reset_index(name='Count')

    print("Label Clustering Statistics:")
    print(count_df)




# 範例用法
csv_file_path = "dataset.csv"  # 替換為你的CSV文件路徑
num_clusters = 2  # 設定聚類數量

# 執行聚類
image_cluster_mapping = cluster_images(csv_file_path, num_clusters)

# 視覺化聚類結果
visualize_clusters(image_cluster_mapping)
