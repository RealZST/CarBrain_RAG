#!/usr/bin/env python
# coding: utf-8

import json
# import jieba
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from langchain.schema import Document
# from langchain.vectorstores import Chroma, FAISS
# from langchain import PromptTemplate, LLMChain
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import time
# import re

from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess


# # 获取Langchain的工具链
# def get_qa_chain(llm, vector_store, prompt_template):
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)


def limit_retrieved_emb_docs(context, top_k=6, max_length=2500):
    '''限制召回的文本块的个数和总长度'''
    emb_ans = ""
    cnt = 0
    for doc, score in context:
        cnt = cnt + 1
        if len(emb_ans + doc.page_content) > max_length:  # 限制召回结果的文本长度
            break
        emb_ans = emb_ans + doc.page_content
        if cnt > top_k:  # 只选前top_k个召回结果
            break
    return emb_ans  # str


def limit_retrieved_docs(context, top_k=6, max_length=2500):
    '''限制召回的文本块的个数和总长度'''
    ans = ""
    cnt = 0
    for doc in context:
        cnt = cnt + 1
        if len(ans + doc.page_content) > max_length:  # 限制召回结果的文本长度
            break
        ans = ans + doc.page_content
        if cnt > top_k:  # 只选前top_k个召回结果
            break
    return ans  # str


def get_emb_bm25_merge(emb_ans, bm25_ans, query):
    '''
    将FAISS 和 BM25 的 召回结果 与 query 组合成 prompt
    '''
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                        如果无法从中得到答案，请说"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                        已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                        1: {emb_ans}
                        2: {bm25_ans}
                        问题:
                        {question}""".format(emb_ans=emb_ans, bm25_ans=bm25_ans, question=query)
    return prompt_template


def get_rerank(emb_ans, query):
    '''
    将 召回结果 与 query 组合成 prompt
    '''
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                        如果无法从中得到答案，请说"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                        已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                        1: {emb_ans}
                        问题:
                        {question}""".format(emb_ans=emb_ans, question=query)
    return prompt_template


# def question(text, llm, vector_store, prompt_template):
#     chain = get_qa_chain(llm, vector_store, prompt_template)
#     response = chain({"query": text})
#     return response


def reRank(rerank, query, faiss_context, bm25_context, top_k, max_length):
    items = []
    for doc, score in faiss_context:
        items.append(doc)
    items.extend(bm25_context)  # items: list[Document]
    rerank_ans = rerank.predict(query, items)
    ans = limit_retrieved_docs(rerank_ans, top_k, max_length)
    return ans  # str


if __name__ == "__main__":

    start = time.time()

    base = "."
    # LLM model
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat/qwen/Qwen-7B-Chat"
    # embedding model
    m3e = base + "/pre_train_model/m3e-large"
    # rerank model
    bge_reranker_large = base + "/pre_train_model/bge-reranker-large"

    # 解析pdf文档，按照不同的规则切分文本
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
    # 将这6种切分结果（3种切分方法，每种方法分别设置2个max_seq值）全部存在data中
    data = dp.data
    print("data load ok")

    # 加载Faiss retriever
    faissretriever = FaissRetriever(m3e, data)
    vector_store = faissretriever.vector_store
    print("faissretriever load ok")

    # 加载BM25 retriever
    bm25 = BM25(data)
    print("bm25 load ok")

    # 加载LLM大模型
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # 加载reRank模型
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")

    # 对测试数据中的每一个问题，生成回答
    with open(base + "/data/test_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(f"Loads {len(jdata)} test questions。")

        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss召回相似文本块
            faiss_context = faissretriever.GetTopK(query, k=15)
            faiss_min_score = 0.0
            if (len(faiss_context) > 0):
                faiss_min_score = faiss_context[0][1]
            emb_ans = limit_retrieved_emb_docs(faiss_context)

            # bm25召回相似文本块
            bm25_context = bm25.GetBM25TopK(query, k=15)
            bm25_ans = limit_retrieved_docs(bm25_context)

            # 将FAISS 和 BM25 的 召回结果 与 query 组合成 prompt
            emb_bm25_merge_inputs = get_emb_bm25_merge(emb_ans, bm25_ans, query)

            # 将faiss召回结果 与 query 组合成 prompt
            emb_inputs = get_rerank(emb_ans, query)

            # 将bm25召回结果 与 query 组合成 prompt
            bm25_inputs = get_rerank(bm25_ans, query)

            # 将FAISS和BM25的召回结果按照与query的相关性得分排序
            rerank_ans = reRank(rerank, query, faiss_context, bm25_context, top_k=6, max_length=4000)
            # 将rerank后的召回结果与 query 组合成 prompt
            rerank_inputs = get_rerank(rerank_ans, query)

            batch_input = []
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(emb_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(rerank_inputs)

            batch_output = llm.infer(batch_input)

            line["answer_1"] = batch_output[0]  # 合并两路召回，得到的answer
            line["answer_2"] = batch_output[1]  # 只考虑faiss召回，得到的answer
            line["answer_3"] = batch_output[2]  # 只考虑bm召回，得到的answer
            line["answer_4"] = batch_output[3]  # 合并两路召回且重排序，得到的answer
            line["answer_5"] = emb_ans  # faiss召回的文本块
            line["answer_6"] = bm25_ans  # bm召回的文本块
            line["answer_7"] = rerank_ans  # 合并两路召回并重排序得到的文本块

            # 如果faiss检索跟query的距离高于500，输出‘无相关文本’
            if faiss_min_score > 500:
                line["answer_5"] = "无相关文本"
            else:
                line["answer_5"] = str(faiss_min_score) + emb_ans

        # 保存结果
        json.dump(jdata, open(base + "/data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))
