"""
Load the Qwen-7B-Chat (Qwen 7B) model using vLLM for batch inference.
"""

import os
import torch
import time

from config import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, get_stop_words_ids


# Disable multi-threading in the transformers tokenizer to reduce multi-threading conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# Stop tokens for the Qwen model
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


# Release unused GPU memory and reduce memory fragmentation
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            # Free GPU memory that is cached but not actively used
            torch.cuda.empty_cache()
            # Clean up CUDA IPC objects to reduce memory fragmentation
            torch.cuda.ipc_collect()


class ChatLLM(object):
    '''
    Wrapper for Qwen-7B-Chat to enable LLM inference.
    Automatically manages the tokenizer, stopwords, GPU memory, etc.
    Supports batch inference.
    '''
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )
        # Load the generation configuration for Qwen-7B-Chat
        self.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=self.tokenizer.pad_token_id)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        self.stop_words_ids = []

        # Load vLLM
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,  # If using multiple GPUs, set this to the number of GPUs (N)
            trust_remote_code=True,
            gpu_memory_utilization=0.6,  # Adjust this based on GPU memory usage
            dtype="bfloat16"
        )

        # Get stopword IDs
        for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

        # Sampling parameters for the LLM
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "early_stopping": False,
            "top_p": 1.0,
            "top_k": -1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
            "temperature": 0.0,
            "max_tokens": 2000,
            "repetition_penalty": self.generation_config.repetition_penalty,
            "n": 1,  # 每次生成 n 个不同的答案并全部返回
            "best_of": 2,  # 生成 best_of 个答案 只返回最好的一个
            "use_beam_search": True
        }
        self.sampling_params = SamplingParams(**sampling_kwargs)

    # Batch inference, input a batch of prompts and return a batch of responses
    def infer(self, prompts):
        batch_text = []
        for q in prompts:
            # make_context() 格式化 Qwen 的输入
            raw_text, _ = make_context(
                self.tokenizer,
                q,
                system="You are a helpful assistant.",
                max_window_size=self.generation_config.max_window_size,
                chat_format=self.generation_config.chat_format,
            )
            batch_text.append(raw_text)

        # Perform batch inference with vLLM
        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)

        # Remove stop tokens
        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            if IMEND in output_str:
                output_str = output_str[:-len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[:-len(ENDOFTEXT)]
            batch_response.append(output_str)

        # Release GPU memory
        torch_gc()

        return batch_response


if __name__ == "__main__":
    qwen7 = "Qwen/Qwen-7B-Chat"
    start = time.time()
    llm = ChatLLM(qwen7)
    test = ["吉利汽车座椅按摩", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("Execution time: " + str((end-start)/60) + " minutes")
