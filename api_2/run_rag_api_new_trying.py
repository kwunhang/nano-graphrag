import asyncio
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from typing_extensions import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from .protocol import ChatCompletionRequest, ChatCompletionResponse
from .chat import (
    create_chat_completion_response,
    create_stream_chat_completion_response,
)

from sse_starlette import EventSourceResponse

# Graph RAG import
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

LLM2_5_BASE_URL = "http://localhost:8000/v1/"
LLM_API_KEY = "sk-22"
MODEL = "DSF-CUG-LLM"
WORKING_DIR = "/ssddata/jimmy/graphrag/Ryan_KG"


# Graph Rag LLM and embedding
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

# Your graphrag class and query function
class MyGaphRAG:
    rag: GraphRAG
    
    def __init__(self, working_dir = WORKING_DIR, llm_model_func=llm_2_5_model_if_cache):
        self.rag = GraphRAG(
        working_dir=working_dir,
        best_model_func=llm_model_func,
        cheap_model_func=llm_model_func,
        embedding_func=azure_openai_embedding,
    )
        
    def run_query(self, query: str, param: QueryParam = QueryParam(mode="local")):
        response =self.rag.query(
            query, param=param
        )
        return response
    
    async def run_aquery(self, query: str, param: QueryParam = QueryParam(mode="local")):
        response = await self.rag.aquery(
            query, param=param
        )
        return response


def create_app(chat_model) -> "FastAPI":
    # Create the FastAPI application
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key verification
    api_key = os.environ.get("API_KEY", None)
    security = HTTPBearer(auto_error=False)

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        if not chat_model:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="No chat model")
        if request.stream:
            generate = create_stream_chat_completion_response(request, chat_model)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            return await create_chat_completion_response(request, chat_model)

    
    return app

# Run the API
def run_api() -> None:
    graphrag = MyGaphRAG(working_dir=WORKING_DIR, llm_model_func=llm_2_5_model_if_cache)

    app = create_app(graphrag)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8002"))
    print(f"Visit http://localhost:{api_port}/docs for API documentation.")
    import uvicorn
    uvicorn.run(app, host=api_host, port=api_port)

if __name__ == "__main__":
    run_api()