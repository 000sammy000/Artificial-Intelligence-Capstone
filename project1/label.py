import os
import pandas as pd

def create_labeled_dataset(image_folder, label):
    df = pd.DataFrame(columns=['Image_Path', 'Label'])  # Change variable name to df

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Check if the file is an image
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            df = df._append({'Image_Path': image_path, 'Label': label}, ignore_index=True)

    return df
    

# Specify folder paths
class1_folder = "data/soccer_resize"
class2_folder = "data/head_resize"

# Create labeled datasets
class1_data = create_labeled_dataset(class1_folder, label=0)
class2_data = create_labeled_dataset(class2_folder, label=1)

# Concatenate the datasets
combined_data = pd.concat([class1_data, class2_data], ignore_index=True)

# Save the dataset to a CSV file
combined_data.to_csv("dataset.csv", index=False)
