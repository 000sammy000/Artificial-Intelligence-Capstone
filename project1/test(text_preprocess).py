import pandas as pd
import jieba

# 創建一個字典，包含文本和標籤
data = {'text': ['這是正面文本1', '這是負面文本2', '這是中性文本3'],
        'label': [1, 0, 2]}

# 將字典轉換為DataFrame
df = pd.DataFrame(data)

# 定義一個函數來應用 jieba 分詞
def chinese_segmentation(text):
    seg_list = jieba.cut(text, cut_all=False)
    return ' '.join(seg_list)

# 將 DataFrame 的 'text' 列應用中文分詞
df['text_seg'] = df['text'].apply(chinese_segmentation)

# 顯示處理後的 DataFrame
print(df)


# 將DataFrame保存為CSV文件
df.to_csv('text_dataset.csv', encoding='utf-8')