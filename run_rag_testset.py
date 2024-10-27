import os
import logging
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
import pandas as pd
from nano_graphrag._llm import get_azure_openai_async_client_instance
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Assumed llm model settings
LLM2_5_BASE_URL = "http://localhost:8000/v1/"
LLM1_5_BASE_URL = "http://localhost:8000/v1/"

LLM_API_KEY = "sk-22"
MODEL = "DSF-CUG-LLM"

# Set directory
WORKING_DIR = "/ssddata/jimmy/graphrag/Ryan_KG"

async def llm_2_5_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM2_5_BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content

async def llm_1_5_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM1_5_BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


class MyGaphRAG:
    rag: GraphRAG
    
    def __init__(self, workding_dir = WORKING_DIR, llm_model_func=llm_2_5_model_if_cache):
        self.rag = GraphRAG(
        working_dir=workding_dir,
        best_model_func=llm_model_func,
        cheap_model_func=llm_model_func,
        embedding_func=azure_openai_embedding,
    )
        
    def run_query(self, query: str, param: QueryParam = QueryParam(mode="local")):
        return self.rag.query(
            query, param=param
        )

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        # model="text-embedding-3-small", input=texts, encoding_format="float"
        model="text-embedding-ada-002", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])



def run_rag_test(test_csv, output_dir, workding_dir = WORKING_DIR):
    test_df = pd.read_csv(test_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # test for 2.5
    graphrag = MyGaphRAG(workding_dir=workding_dir, llm_model_func=llm_2_5_model_if_cache)
    results = []
    for _, cur_row in test_df.iterrows():
        input_text = cur_row['Question']
        print(input_text)
        output = graphrag.run_query(input_text)
        print(output)
        results.append({"Question": input_text, "LLM Answer": output})
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "prompt1-sft-Qwen2.5-运维知识-answer.csv")
    results_df.to_csv(output_file, index=False)

    # test for 1.5
    graphrag = MyGaphRAG(workding_dir=workding_dir, llm_model_func=llm_1_5_model_if_cache)
    results = []
    for _, cur_row in test_df.iterrows():
        input_text = cur_row['Question']
        print(input_text)
        output = graphrag.run_query(input_text)
        print(output)
        results.append({"Question": input_text, "LLM Answer": output})
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "prompt1-sft-Qwen1.5-运维知识-answer.csv")
    results_df.to_csv(output_file, index=False)



if __name__=="__main__":
    test_csv = "/ssddata/jimmy/data/运维知识.csv"
    output_dir = "/ssddata/jimmy/result/graphrag"
    run_rag_test(test_csv, output_dir)



