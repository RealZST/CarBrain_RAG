"""
接收faiss和bm25检索得到的文本块，计算与query的相关性scores，按 scores 降序排列文本块并返回。
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

# from bm25_retriever import BM25
# from pdf_parse import DataProcess
from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            # 释放未被占用但仍然缓存的 GPU 显存
            torch.cuda.empty_cache()
            # 清理 CUDA IPC（进程间通信）对象，减少显存碎片
            torch.cuda.ipc_collect()


# 加载rerank模型
class reRankLLM(object):
    def __init__(self, model_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()  # 使用 FP16降低显存占用
        # self.model.cuda()
        self.model.to(CUDA_DEVICE)
        self.max_length = max_length  # 最长输入序列长度

    def predict(self, query, docs):
        '''
        输入 query 和 docs（候选文档列表）
        计算 (query, doc) 相关性得分
        按分数降序排序，返回最相关的文档
        '''
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(
            pairs, 
            padding=True,  # 批量输入（batch）时，要保证所有句子等长
            truncation=True,  # 如果 query + doc 过长，截断到 max_length=512
            return_tensors='pt', 
            max_length=self.max_length
        ).to("cuda")
        # 计算相关性得分
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        # 按 scores 降序排列 docs
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x:x[0])]
        # 释放显存
        torch_gc()
        
        return response


if __name__ == "__main__":
    bge_reranker_large = "./pre_train_model/bge-reranker-large"
    rerank = reRankLLM(bge_reranker_large)
