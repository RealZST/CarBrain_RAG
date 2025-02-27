"""
Receive text chunks retrieved from FAISS and BM25,
calculate their relevance scores with the query,
sort the text chunks in descending order based on scores, and return them.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# Release unused GPU memory and reduce memory fragmentation
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            # Release GPU memory that is cached but not actively used
            torch.cuda.empty_cache()
            # Clean up CUDA IPC objects to reduce memory fragmentation
            torch.cuda.ipc_collect()


# Creat the rerank model
class reRankLLM(object):
    def __init__(self, model_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()  # FP16
        self.model.to(CUDA_DEVICE)
        self.max_length = max_length  # Maximum input sequence length

    def predict(self, query, docs):
        '''
        Input: query and docs (a list of retrieved documents)
        Compute relevance scores for (query, doc) pairs
        Sort the documents in descending order based on scores and return the most relevant ones
        '''
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,  # Ensure all sentences are of equal length in batch processing
            truncation=True,  # Truncate if query + doc exceeds max_length
            return_tensors='pt',
            max_length=self.max_length
        ).to("cuda")
        # Compute relevance scores
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        # Sort docs in descending order based on scores
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x:x[0])]
        # Release GPU memory
        torch_gc()

        return response


if __name__ == "__main__":
    bge_reranker_large = "BAAI/bge-reranker-large"
    rerank = reRankLLM(bge_reranker_large)
