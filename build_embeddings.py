import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# 配置
DATA_FOLDER = './data'
EMBEDDING_FOLDER = './embeddings'
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'  # 您也可以选择其他模型

def build_embeddings():
    # 创建嵌入存储文件夹
    if not os.path.exists(EMBEDDING_FOLDER):
        os.makedirs(EMBEDDING_FOLDER)

    # 加载模型
    model = SentenceTransformer(MODEL_NAME)

    # 读取所有文档
    doc_texts = []
    doc_ids = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.txt'):
            filepath = os.path.join(DATA_FOLDER, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                doc_texts.append(text)
                doc_ids.append(filename)

    # 生成嵌入
    print('正在生成嵌入...')
    embeddings = model.encode(doc_texts, show_progress_bar=True)

    # 保存嵌入和文档ID
    np.save(os.path.join(EMBEDDING_FOLDER, 'doc_embeddings.npy'), embeddings)
    with open(os.path.join(EMBEDDING_FOLDER, 'doc_ids.pkl'), 'wb') as f:
        pickle.dump(doc_ids, f)

    print('嵌入已生成并保存。')

if __name__ == '__main__':
    build_embeddings()