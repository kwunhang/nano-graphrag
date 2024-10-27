import os
import logging
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
import asyncio
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
LLM_BASE_URL = "http://localhost:8000/v1/"
LLM_API_KEY = "sk-22"
MODEL = "DSF-CUG-LLM"

# Set directory
WORKING_DIR = "/ssddata/jimmy/graphrag/Error_Product_KG"

async def llm_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
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

def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        # embedding_func=azure_openai_embedding,
    )
    print(
        "#########\n",
        rag.query(
            "CloudSE2980返回404，提示“Find User By INFO Xml C-MSISDN Failed”", param=QueryParam(mode="local")
        )
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


if __name__=="__main__":
    query()



# # Assumed llm model 
# graph_func = GraphRAG(working_dir="/ssddata/jimmy/error_product_kg/test_KG")

# with open("/ssddata/jimmy/error_product_kg/test_KG/book.txt") as f:
#     graph_func.insert(f.read())

# # Perform global graphrag search
# print(graph_func.query("What are the top themes in this story?"))

# # Perform local graphrag search (I think is better and more scalable one)
# print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))