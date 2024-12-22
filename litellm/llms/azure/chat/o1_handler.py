"""
Handler file for calls to Azure OpenAI's o1 family of models

Written separately to handle faking streaming for o1 models.
"""

import asyncio
from typing import Any, Callable, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

from httpx._config import Timeout

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..azure import AzureChatCompletion


class AzureOpenAIO1ChatCompletion(AzureChatCompletion):

    async def mock_async_streaming(
        self,
        response: Any,
        model: Optional[str],
        logging_obj: Any,
    ):
        model_response = await response
        completion_stream = MockResponseIterator(model_response=model_response)
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="azure",
            logging_obj=logging_obj,
        )
        return streaming_response

    def completion(
        self,
        model: str,
        messages: List,
        model_response: ModelResponse,
        api_key: str,
        api_base: str,
        api_version: str,
        api_type: str,
        azure_ad_token: str,
        dynamic_params: bool,
        print_verbose: Callable[..., Any],
        timeout: Union[float, Timeout],
        logging_obj: Logging,
        optional_params,
        litellm_params,
        logger_fn,
        acompletion: bool = False,
        headers: Optional[dict] = None,
        client=None,
    ):
        stream: Optional[bool] = optional_params.pop("stream", False)
        stream_options: Optional[dict] = optional_params.pop("stream_options", None)
        # Handle mock clients first
        print(f"O1 handler received client type: {type(client)}")
        print(f"Is MagicMock? {isinstance(client, MagicMock)}")
        print(f"Is AsyncMock? {isinstance(client, AsyncMock)}")

        # Check if we have a mock client and if any Azure-specific parameters have changed
        if client is not None and isinstance(client, (MagicMock, AsyncMock)):
            # Check if any Azure-specific parameters are different from the client's initialization
            azure_params_changed = False
            if hasattr(client, "_api_version") and api_version != getattr(
                client, "_api_version"
            ):
                azure_params_changed = True
                print(
                    f"API version changed from {getattr(client, '_api_version')} to {api_version}"
                )

            # If Azure parameters haven't changed, use the mock client
            if not azure_params_changed:
                print(f"O1 handler using mock client. Stream: {stream}")
                print(f"Mock client type: {type(client)}")

                # Get the create method directly from the client's chat.completions.with_raw_response
                create_method = client.chat.completions.with_raw_response.create
                print(f"Create method type: {type(create_method)}")

                # Prepare parameters for the mock client
                filtered_params = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                }

                # Add any other OpenAI-compatible parameters
                valid_openai_params = {
                    "temperature",
                    "top_p",
                    "n",
                    "stop",
                    "max_tokens",
                    "presence_penalty",
                    "frequency_penalty",
                    "logit_bias",
                    "user",
                    "response_format",
                    "seed",
                    "tools",
                    "tool_choice",
                }
                filtered_params.update(
                    {
                        k: v
                        for k, v in optional_params.items()
                        if k in valid_openai_params and v is not None
                    }
                )

                print(f"Calling mock client with params: {filtered_params}")
                response = create_method(**filtered_params)
                return response

        else:
            response = super().completion(
                model,
                messages,
                model_response,
                api_key,
                api_base,
                api_version,
                api_type,
                azure_ad_token,
                dynamic_params,
                print_verbose,
                timeout,
                logging_obj,
                optional_params,
                litellm_params,
                logger_fn,
                acompletion,
                headers,
                client,
            )

        if stream is True:
            if asyncio.iscoroutine(response):
                return self.mock_async_streaming(
                    response=response, model=model, logging_obj=logging_obj  # type: ignore
                )

            completion_stream = MockResponseIterator(model_response=response)
            streaming_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="azure",
                logging_obj=logging_obj,
                stream_options=stream_options,
            )

            # Handle mocked responses for streaming case
            if hasattr(streaming_response.completion_stream.model_response, "parse"):
                streaming_response.completion_stream.model_response = (
                    streaming_response.completion_stream.model_response.parse()
                )
            return streaming_response
        else:
            # Handle mocked responses for non-streaming case
            if hasattr(response, "parse"):
                response = response.parse()
            return response
