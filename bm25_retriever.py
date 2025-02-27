#!/usr/bin/env python
# coding: utf-8

"""
Create BM25 retriever
Tokenize the query and retrieve the top k most similar text blocks
"""

from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba


class BM25(object):
    # Pass in a list of pre-segmented text blocks, iterate through the list,
    # perform tokenization, and establish an index and mapping relationship 
    # between tokenized documents and full-text documents
    def __init__(self, documents):
        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip()
            # Skip if the line is too short (less than 5 characters)
            if len(line) < 5:
                continue
            # Perform Chinese word segmentation on the line
            # and join the segmented words with a space
            tokens = " ".join(jieba.cut_for_search(line))
            # Store the tokenized text and document ID
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            words = line.split("\t")
            # Store the original text (full-text) and document ID
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        # BM25 requires tokenized text to calculate similarity
        self.documents = docs
        # The original text (full-text) is needed when returning documents
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    # Initialize BM25 retriever
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # Get the top k documents based on BM25 scores
    def GetBM25TopK(self, query, k):
        self.retriever.k = k
        # Tokenize the query
        query = " ".join(jieba.cut_for_search(query))
        # Retrieve similar documents (BM25 does not explicitly provide scores)
        ans_docs = self.retriever.get_relevant_documents(query)
        # Map back to the original text and return
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])

        return ans  # list[Document]


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
