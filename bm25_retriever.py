#!/usr/bin/env python
# coding: utf-8

"""
创建 bm25 retriever
对 query 进行分词，获取相似度最高的 k 个文本块
"""

from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba


class BM25(object):
    # 传入被切分好的文本块列表，遍历文本块列表，首先做分词，然后把分词后的文档和全文文档建立索引和映射关系
    def __init__(self, documents):
        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip()
            # 如果 line 过短（少于 5 个字符），则跳过
            if len(line) < 5:
                continue
            # 对line 进行中文分词，将分词结果用 " " 连接成字符串
            tokens = " ".join(jieba.cut_for_search(line))
            # 存储 分词后的文本和文档 ID
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            words = line.split("\t")
            # 存储 原始文本和文档 ID
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        # BM25 计算相似度时需要分词后的文本
        self.documents = docs
        # 返回文档时需要原始文本
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    # 初始化BM25 retriever
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在top k的文档和分数
    def GetBM25TopK(self, query, k):
        self.retriever.k = k
        # 对 query 进行分词
        query = " ".join(jieba.cut_for_search(query))
        # 获取相似文档(BM25 不显示提供score)
        ans_docs = self.retriever.get_relevant_documents(query)
        # 索引原始文本并返回
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        # ans 是list[Document]
        return ans


if __name__ == "__main__":
    dp = DataProcess(pdf_path="./data/train_a.pdf")
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

    bm25 = BM25(data)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)
