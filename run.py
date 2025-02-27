#!/usr/bin/env python
# coding: utf-8

import json
# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
import time
from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess


# def get_qa_chain(llm, vector_store, prompt_template):
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)

# Limit the number of retrieved text blocks and their total length
def limit_retrieved_emb_docs(context, top_k=6, max_length=2500):
    emb_ans = ""
    cnt = 0
    for doc, score in context:
        cnt = cnt + 1
        if len(emb_ans + doc.page_content) > max_length:  # Restrict the total length of retrieved results
            break
        emb_ans = emb_ans + doc.page_content
        if cnt > top_k:  # Select only the top_k retrieved results
            break
    return emb_ans  # str


# Limit the number of retrieved text blocks and their total length
def limit_retrieved_docs(context, top_k=6, max_length=2500):
    ans = ""
    cnt = 0
    for doc in context:
        cnt = cnt + 1
        if len(ans + doc.page_content) > max_length:  # Restrict the total length of retrieved results
            break
        ans = ans + doc.page_content
        if cnt > top_k:  # Select only the top_k retrieved results
            break
    return ans  # str


# Merge FAISS and BM25 retrieval results with the query into a prompt
def get_emb_bm25_merge(emb_ans, bm25_ans, query):
    prompt_template = """基于以下已知信息，用中文简洁和专业地来回答用户的问题。
                        如果 **无法从中得到答案**，必须仅输出 **“无答案”**（不允许任何其他内容）。
                        请 **先判断相关性**，如果无关，则必须严格输出 **“无答案”**。
                        已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                        1: {emb_ans}
                        2: {bm25_ans}
                        问题:
                        {question}""".format(emb_ans=emb_ans, bm25_ans=bm25_ans, question=query)
    return prompt_template


# Merge retrieved results with the query into a prompt
def get_rerank(emb_ans, query):
    prompt_template = """基于以下已知信息，用中文简洁和专业地来回答用户的问题。
                        如果 **无法从中得到答案**，必须仅输出 **“无答案”**（不允许任何其他内容）。
                        请 **先判断相关性**，如果无关，则必须严格输出 **“无答案”**。
                        已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                        {emb_ans}
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

    # LLM model
    qwen7 = "Qwen/Qwen-7B-Chat"

    # embedding model
    m3e = "moka-ai/m3e-large"

    # rerank model
    bge_reranker_large = "BAAI/bge-reranker-large"

    # Parse the PDF document and segment text using different rules
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
    # Store all six segmentation results in data (3 segmentation methods, each with 2 max_seq values) 
    data = dp.data
    print("data load ok")

    # Load FAISS retriever
    faissretriever = FaissRetriever(m3e, data)
    vector_store = faissretriever.vector_store
    print("faissretriever load ok")

    # Load BM25 retriever
    bm25 = BM25(data)
    print("bm25 load ok")

    # Load LLM model
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # Load reRank model
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")

    # Generate answers for each question in the test dataset
    with open("./data/test_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(f"Loads {len(jdata)} test questions。")

        for idx, line in enumerate(jdata):
            query = line["question"]

            # Retrieve similar docs using FAISS
            faiss_context = faissretriever.GetTopK(query, k=15)
            faiss_min_score = 0.0
            if (len(faiss_context) > 0):
                faiss_min_score = faiss_context[0][1]
            emb_ans = limit_retrieved_emb_docs(faiss_context)

            # Retrieve similar docs using BM25
            bm25_context = bm25.GetBM25TopK(query, k=15)
            bm25_ans = limit_retrieved_docs(bm25_context)

            # Merge FAISS and BM25 retrieval results with the query into a prompt
            emb_bm25_merge_inputs = get_emb_bm25_merge(emb_ans, bm25_ans, query)

            # Merge FAISS retrieval results with the query into a prompt
            emb_inputs = get_rerank(emb_ans, query)

            # Merge BM25 retrieval results with the query into a prompt
            bm25_inputs = get_rerank(bm25_ans, query)

            # Re-rank the FAISS and BM25 retrieval results based on relevance scores
            rerank_ans = reRank(rerank, query, faiss_context, bm25_context, top_k=6, max_length=4000)
            # Merge re-ranked retrieval results with the query into a prompt
            rerank_inputs = get_rerank(rerank_ans, query)

            batch_input = [emb_bm25_merge_inputs, emb_inputs, bm25_inputs, rerank_inputs]
            batch_output = llm.infer(batch_input)

            line["answer_1"] = batch_output[0].strip()  # Answer considering both FAISS and BM25 retrieval
            line["answer_2"] = batch_output[1].strip()  # Answer considering only FAISS retrieval
            line["answer_3"] = batch_output[2].strip()  # Answer considering only BM25 retrieval
            line["answer_4"] = batch_output[3].strip()  # Answer considering both retrieval methods and re-ranking
            line["answer_5"] = emb_ans  # FAISS retrieved text blocks
            line["answer_6"] = bm25_ans  # BM25 retrieved text blocks
            line["answer_7"] = rerank_ans  # Merged and re-ranked retrieval text blocks

            # If FAISS retrieval distance is greater than 500, output 'No relevant text'
            if faiss_min_score > 500:
                line["answer_5"] = "无相关文本"
            else:
                line["answer_5"] = str(faiss_min_score) + emb_ans

        # Save the results
        json.dump(jdata, open("./data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))
