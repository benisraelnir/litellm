"""Mock completion utilities for LiteLLM.

This module provides mock completion functionality for testing and debugging purposes.
It includes functions for generating mock responses in both streaming and non-streaming formats.
"""

import asyncio
import time
from typing import Any, List, Optional, Union

import httpx
import openai

import litellm
from litellm.types.utils import ChatCompletionMessageToolCall
from litellm.utils import Choices, CustomStreamWrapper, Message, ModelResponse, Usage


def mock_completion_streaming_obj(model_response, mock_response, model, n=None):
    """Generate a mock streaming completion object.

    Args:
        model_response (ModelResponse): The base model response object.
        mock_response (str): The mock response content.
        model (str): The model identifier.
        n (Optional[int]): Number of completion choices to generate.

    Yields:
        ModelResponse: Streaming response chunks.
    """
    try:
        content = ""
        chunk_size = 4
        response_array = []

        if isinstance(mock_response, str):
            response_array = [
                mock_response[i : i + chunk_size]
                for i in range(0, len(mock_response), chunk_size)
            ]
        else:
            response_array = [mock_response]

        for i, chunk in enumerate(response_array):
            content += str(chunk)
            delta = (
                {"content": str(chunk)}
                if i != len(response_array) - 1
                else {"content": str(chunk), "role": "assistant"}
            )

            if n and n > 1:
                choices = []
                for j in range(n):
                    choice = Choices(
                        index=j,
                        delta=delta,
                        finish_reason="stop" if i == len(response_array) - 1 else None,
                    )
                    choices.append(choice)
                model_response.choices = choices
            else:
                model_response.choices[0].delta = delta
                model_response.choices[0].finish_reason = (
                    "stop" if i == len(response_array) - 1 else None
                )

            model_response.model = model
            yield model_response
            time.sleep(0.02)
    except Exception as e:
        raise Exception(f"Mock streaming failed: {str(e)}")


async def async_mock_completion_streaming_obj(
    model_response, mock_response, model, n=None
):
    """Generate an async mock streaming completion object.

    Args:
        model_response (ModelResponse): The base model response object.
        mock_response (str): The mock response content.
        model (str): The model identifier.
        n (Optional[int]): Number of completion choices to generate.

    Yields:
        ModelResponse: Streaming response chunks.
    """
    try:
        content = ""
        chunk_size = 4
        response_array = []

        if isinstance(mock_response, str):
            response_array = [
                mock_response[i : i + chunk_size]
                for i in range(0, len(mock_response), chunk_size)
            ]
        else:
            response_array = [mock_response]

        for i, chunk in enumerate(response_array):
            content += str(chunk)
            delta = (
                {"content": str(chunk)}
                if i != len(response_array) - 1
                else {"content": str(chunk), "role": "assistant"}
            )

            if n and n > 1:
                choices = []
                for j in range(n):
                    choice = Choices(
                        index=j,
                        delta=delta,
                        finish_reason="stop" if i == len(response_array) - 1 else None,
                    )
                    choices.append(choice)
                model_response.choices = choices
            else:
                model_response.choices[0].delta = delta
                model_response.choices[0].finish_reason = (
                    "stop" if i == len(response_array) - 1 else None
                )

            model_response.model = model
            yield model_response
            await asyncio.sleep(0.02)
    except Exception as e:
        raise Exception(f"Async mock streaming failed: {str(e)}")


def mock_completion(
    model: str,
    messages: List,
    stream: Optional[bool] = False,
    n: Optional[int] = None,
    mock_response: Union[str, Exception, dict] = "This is a mock request",
    mock_tool_calls: Optional[List] = None,
    logging: Optional[Any] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Any:
    """Generate a mock completion response for testing or debugging purposes.

    This function simulates the response structure of the OpenAI completion API.

    Args:
        model: The name of the language model for which the mock response is generated.
        messages: A list of message objects representing the conversation context.
        stream: If True, returns a mock streaming response.
        n: Number of completion choices to generate.
        mock_response: The content of the mock response.
        mock_tool_calls: Optional list of tool calls to include in the response.
        logging: Optional logging object for tracking requests.
        custom_llm_provider: Optional custom LLM provider name.
        **kwargs: Additional keyword arguments.

    Returns:
        ModelResponse: A response object simulating a completion response.

    Raises:
        Exception: If an error occurs during mock response generation.
        litellm.MockException: For simulated API errors.
        litellm.RateLimitError: For simulated rate limit errors.
        litellm.InternalServerError: For simulated server errors.
    """
    try:
        ## LOGGING
        if logging is not None:
            logging.pre_call(
                input=messages,
                api_key="mock-key",
            )
        if isinstance(mock_response, Exception):
            if isinstance(mock_response, openai.APIError):
                raise mock_response
            raise litellm.MockException(
                status_code=getattr(mock_response, "status_code", 500),
                message=getattr(mock_response, "text", str(mock_response)),
                llm_provider=getattr(
                    mock_response, "llm_provider", custom_llm_provider or "openai"
                ),
                model=model,
                request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
            )
        elif (
            isinstance(mock_response, str) and mock_response == "litellm.RateLimitError"
        ):
            raise litellm.RateLimitError(
                message="this is a mock rate limit error",
                llm_provider=getattr(
                    mock_response, "llm_provider", custom_llm_provider or "openai"
                ),
                model=model,
            )
        elif (
            isinstance(mock_response, str)
            and mock_response == "litellm.InternalServerError"
        ):
            raise litellm.InternalServerError(
                message="this is a mock internal server error",
                llm_provider=getattr(
                    mock_response, "llm_provider", custom_llm_provider or "openai"
                ),
                model=model,
            )
        elif isinstance(mock_response, str) and mock_response.startswith(
            "Exception: content_filter_policy"
        ):
            raise litellm.MockException(
                status_code=400,
                message=mock_response,
                llm_provider="azure",
                model=model,
                request=httpx.Request(method="POST", url="https://api.openai.com/v1/"),
            )
        elif isinstance(mock_response, str) and mock_response.startswith(
            "Exception: mock_streaming_error"
        ):
            mock_response = litellm.MockException(
                message="This is a mock error raised mid-stream",
                llm_provider="anthropic",
                model=model,
                status_code=529,
            )

        time_delay = kwargs.get("mock_delay", None)
        if time_delay is not None:
            time.sleep(time_delay)

        if isinstance(mock_response, dict):
            return ModelResponse(**mock_response)

        model_response = ModelResponse(stream=stream)
        if stream is True:
            # don't try to access stream object,
            if kwargs.get("acompletion", False) is True:
                return CustomStreamWrapper(
                    completion_stream=async_mock_completion_streaming_obj(
                        model_response, mock_response=mock_response, model=model, n=n
                    ),
                    model=model,
                    custom_llm_provider="openai",
                    logging_obj=logging,
                )
            return CustomStreamWrapper(
                completion_stream=mock_completion_streaming_obj(
                    model_response, mock_response=mock_response, model=model, n=n
                ),
                model=model,
                custom_llm_provider="openai",
                logging_obj=logging,
            )

        if isinstance(mock_response, litellm.MockException):
            raise mock_response

        if n is None:
            model_response.choices[0].message.content = mock_response
        else:
            _all_choices = []
            for i in range(n):
                _choice = Choices(
                    index=i,
                    message=Message(content=mock_response, role="assistant"),
                )
                _all_choices.append(_choice)
            model_response.choices = _all_choices

        model_response.created = int(time.time())
        model_response.model = model

        if mock_tool_calls:
            model_response.choices[0].message.tool_calls = [
                ChatCompletionMessageToolCall(**tool_call)
                for tool_call in mock_tool_calls
            ]

        setattr(
            model_response,
            "usage",
            Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        try:
            _, custom_llm_provider, _, _ = litellm.utils.get_llm_provider(model=model)
            model_response._hidden_params["custom_llm_provider"] = custom_llm_provider
        except Exception:
            # dont let setting a hidden param block a mock_respose
            pass

        if logging is not None:
            logging.post_call(
                input=messages,
                api_key="my-secret-key",
                original_response="my-original-response",
            )
        return model_response

    except Exception as e:
        if isinstance(e, openai.APIError):
            raise e
        raise Exception("Mock completion response failed")
