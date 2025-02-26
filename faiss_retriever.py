#!/usr/bin/env python
# coding: utf-8

"""
创建FAISS vectordb
对 query 进行向量化，获取相似度最高的 k 个文本块
"""

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch


class FaissRetriever(object):
    # 创建FAISS vectordb
    # 传入embedding model路径，和被切分好的文本块列表
    def __init__(self, model_path, data):
        self.embeddings = HuggingFaceEmbeddings(
                               model_name=model_path,
                               model_kwargs={"device": "cuda"},
                               encode_kwargs={"batch_size": 64}
                           )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

        # # 生成embeddings的时间较长，跑完第一次可以把结果持久化，后面直接load
        # self.vector_store.save_local("./faiss_index")
        # self.vector_store = FAISS.load_local("./faiss_index", self.embeddings, allow_dangerous_deserialization=True)

        # 清理 GPU 显存
        del self.embeddings
        torch.cuda.empty_cache()

    # 对 query 进行向量化，获取相似度最高的 k 个文本块
    def GetTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        # context 是list[(Document, score)]
        return context

    # 返回 FAISS 向量数据库
    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    base = "."
    model_name = base + "/pre_train_model/m3e-large"
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)

    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)

    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)

    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)

    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)
