import base64
import io
import json
import os
import re
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from fastapi import HTTPException, status
from common import dictify, jsonify
import logging
from PIL import Image


from protocol import Role as DataRole
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    Finish,
    Function,
    FunctionCall,
    Role,
)

logger = logging.getLogger("nano-graphrag").setLevel(logging.INFO)

ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}


def _process_request(
    request: "ChatCompletionRequest",
) -> Tuple[List[Dict[str, str]], Optional[str], Optional[str], Optional["ImageInput"]]:
    logger.info("==== request ====\n{}".format(json.dumps(dictify(request), indent=2, ensure_ascii=False)))

    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

    if request.messages[0].role == Role.SYSTEM:
        system = request.messages.pop(0).content
    else:
        system = None

    if len(request.messages) % 2 == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

    input_messages = []
    image = None
    for i, message in enumerate(request.messages):
        if i % 2 == 0 and message.role not in [Role.USER, Role.TOOL]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
        elif i % 2 == 1 and message.role not in [Role.ASSISTANT, Role.FUNCTION]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")

        if message.role == Role.ASSISTANT and isinstance(message.tool_calls, list) and len(message.tool_calls):
            tool_calls = [
                {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                for tool_call in message.tool_calls
            ]
            content = json.dumps(tool_calls, ensure_ascii=False)
            input_messages.append({"role": ROLE_MAPPING[Role.FUNCTION], "content": content})
        elif isinstance(message.content, list):
            for input_item in message.content:
                if input_item.type == "text":
                    input_messages.append({"role": ROLE_MAPPING[message.role], "content": input_item.text})
                else:
                    image_url = input_item.image_url.url
                    if re.match(r"^data:image\/(png|jpg|jpeg|gif|bmp);base64,(.+)$", image_url):  # base64 image
                        image_stream = io.BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1]))
                    elif os.path.isfile(image_url):  # local file
                        image_stream = open(image_url, "rb")
                    else:  # web uri
                        image_stream = requests.get(image_url, stream=True).raw

                    image = Image.open(image_stream).convert("RGB")
        else:
            input_messages.append({"role": ROLE_MAPPING[message.role], "content": message.content})

    tool_list = request.tools
    if isinstance(tool_list, list) and len(tool_list):
        try:
            tools = json.dumps([dictify(tool.function) for tool in tool_list], ensure_ascii=False)
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
    else:
        tools = None

    return input_messages, system, tools, image


def _create_stream_chat_completion_chunk(
    completion_id: str,
    model: str,
    delta: "ChatCompletionMessage",
    index: Optional[int] = 0,
    finish_reason: Optional["Finish"] = None,
) -> str:
    choice_data = ChatCompletionStreamResponseChoice(index=index, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionStreamResponse(id=completion_id, model=model, choices=[choice_data])
    return jsonify(chunk)


async def create_stream_chat_completion_response(
    request: "ChatCompletionRequest", chat_model: "ChatModel"
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    input_messages, system, tools, image = _process_request(request)

    if tools:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

    if request.n > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream multiple responses.")

    # Start with an empty message
    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(role=Role.ASSISTANT, content="")
    )

    async for new_token in chat_model.astream_chat(
        input_messages,
        system,
        tools,
        image,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        stop=request.stop,
    ):
        if new_token:
            yield _create_stream_chat_completion_chunk(
                completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(content=new_token)
            )

    # Finish the stream
    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    yield "[DONE]"

async def create_chat_completion_response(
    request: "ChatCompletionRequest", chat_model
) -> "ChatCompletionResponse":
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    input_messages, system, tools, image = _process_request(request)

    responses = await chat_model.rag.aquery(
        input_messages,
        system,
        tools,
        image,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        num_return_sequences=request.n,
        stop=request.stop,
    )

    prompt_length, response_length = 0, 0
    choices = []
    
    for i, response in enumerate(responses):
        result = response.response_text if not tools else chat_model.rag.engine.template.extract_tool(response.response_text)

        if isinstance(result, list):
            tool_calls = [
                FunctionCall(id=f"call_{uuid.uuid4().hex}", function=Function(name=tool[0], arguments=tool[1]))
                for tool in result
            ]
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=tool_calls)
            finish_reason = Finish.TOOL
        else:
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, content=result)
            finish_reason = Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH

        choices.append(ChatCompletionResponseChoice(index=i, message=response_message, finish_reason=finish_reason))
        prompt_length = response.prompt_length
        response_length += response.response_length

    usage = ChatCompletionResponseUsage(
        prompt_tokens=prompt_length,
        completion_tokens=response_length,
        total_tokens=prompt_length + response_length,
    )

    return ChatCompletionResponse(id=completion_id, model=request.model, choices=choices, usage=usage)