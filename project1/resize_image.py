import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, new_size):
    """
    批量調整資料夾中所有圖片的大小

    Parameters:
    - input_folder: 輸入圖片所在的資料夾路徑
    - output_folder: 調整大小後的圖片輸出資料夾路徑
    - new_size: 新的大小，以元組 (width, height) 表示
    """
    try:
        # 如果輸出資料夾不存在，則創建它
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍歷輸入資料夾中的所有檔案
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)

            # 檢查檔案是否為圖片
            if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 打開圖片
                with Image.open(input_path) as img:
                    # 調整大小
                    resized_img = img.resize(new_size)
                    
                    # 生成輸出路徑
                    output_path = os.path.join(output_folder, filename)
                    
                    # 保存新圖片
                    resized_img.save(output_path)
                    print(f"已處理: {filename}")

        print("所有圖片調整大小成功！")
    except Exception as e:
        print(f"處理圖片時發生錯誤：{e}")

# 範例用法
input_folder_path = "data/head_cut"
output_folder_path = "data/head_resize"
new_size = (100, 100)  # 設定新的大小，例如 (width, height)

resize_images_in_folder(input_folder_path, output_folder_path, new_size)
