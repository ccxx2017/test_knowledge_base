import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 配置零一万物 API
API_KEY = os.getenv("API_KEY")
API_BASE = "https://api.lingyiwanwu.com/v1"

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# 配置
EMBEDDING_FOLDER = './embeddings'
DATA_FOLDER = './data'
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'  # 与之前使用的模型一致
TOP_K = 3  # 选择与查询最相似的前K个文档

def load_embeddings():
    embeddings = np.load(os.path.join(EMBEDDING_FOLDER, 'doc_embeddings.npy'))
    with open(os.path.join(EMBEDDING_FOLDER, 'doc_ids.pkl'), 'rb') as f:
        doc_ids = pickle.load(f)
    return embeddings, doc_ids

def semantic_search(query, embeddings, doc_ids):
    # 加载模型
    model = SentenceTransformer(MODEL_NAME)

    # 对查询进行编码
    query_embedding = model.encode([query])

    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # 获取最相似的文档索引
    top_k_indices = similarities.argsort()[-TOP_K:][::-1]

    # 获取对应的文档内容
    retrieved_docs = []
    for idx in top_k_indices:
        doc_id = doc_ids[idx]
        doc_path = os.path.join(DATA_FOLDER, doc_id)
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
            retrieved_docs.append(text)
    return retrieved_docs

def generate_answer(query, retrieved_docs):
    # 构建提示词
    context = "\n\n".join(retrieved_docs)
    prompt = f"""请根据以下**背景知识**来回答问题。如果背景知识中有相关内容，请尽量引用并解释。：

背景知识：
{context}

问题：
{query}

答案：
"""

    # 使用新版本 OpenAI API 调用 Yi-Large 模型生成回答
    response = client.chat.completions.create(
        model="yi-large",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
        n=1,
    )

    answer = response.choices[0].message.content.strip()
    return answer

def main():
    # 加载嵌入
    embeddings, doc_ids = load_embeddings()

    while True:
        query = input('请输入您的问题（输入"退出"结束）：')
        if query == '退出':
            break

        # 语义检索
        retrieved_docs = semantic_search(query, embeddings, doc_ids)
        # **打印检索到的文档内容**
        # print("\n检索到的文档内容：")
        # for idx, doc in enumerate(retrieved_docs):
        #     print(f"文档 {idx+1}：\n{doc}\n")

        # 生成回答
        answer = generate_answer(query, retrieved_docs)

        print(f'\n回答：\n{answer}\n')

if __name__ == '__main__':
    main()