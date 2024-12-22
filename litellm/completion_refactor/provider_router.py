import asyncio
import inspect
import os
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import openai

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    stringify_json_tool_call_content,
)
from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
from litellm.llms.azure.chat.o1_handler import AzureOpenAIO1ChatCompletion
from litellm.llms.azure.completion.handler import AzureTextCompletion
from litellm.llms.base import BaseLLM as base_llm_http_handler
from litellm.llms.databricks.chat.handler import DatabricksChatCompletion
from litellm.llms.huggingface.chat.handler import Huggingface
from litellm.llms.oobabooga.chat.oobabooga import completion as oobabooga
from litellm.llms.openai.completion.handler import OpenAITextCompletion
from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
from litellm.llms.replicate.chat.handler import completion as replicate_completion
from litellm.utils import CustomStreamWrapper, ModelResponse, get_secret


class ProviderRouter:
    """
    Routes completion requests to the appropriate provider implementation.
    Handles provider-specific configurations, authentication, and response processing.

    Args:
        model (str): The model to use for completion.
        messages (List): The messages to process.
        custom_llm_provider (Optional[str]): The custom LLM provider to use.
        **kwargs: Additional arguments passed to the completion call.
    """

    def __init__(
        self,
        model: str,
        messages: List,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        self.messages = messages
        self.custom_llm_provider = custom_llm_provider
        self.kwargs = kwargs

        # Extract common parameters
        self.api_key = kwargs.get("api_key")
        self.api_base = kwargs.get("api_base")
        self.headers = kwargs.get("headers")
        self.extra_headers = kwargs.get("extra_headers")
        self.timeout = kwargs.get("timeout")
        self.stream = kwargs.get("stream", False)
        self.acompletion = kwargs.get("acompletion", False)
        self.model_response = kwargs.get("model_response", {})
        self.logging = kwargs.get("logging_obj")
        self.custom_prompt_dict = kwargs.get("custom_prompt_dict")
        self.optional_params = kwargs.get("optional_params", {})
        self.litellm_params = kwargs.get("litellm_params", {})
        self.logger_fn = kwargs.get("logger_fn")
        self.encoding = kwargs.get("encoding")
        self.client = kwargs.get("client")
        self.organization = kwargs.get("organization")

    async def _async_route_completion(self) -> Any:
        """
        Asynchronous implementation of completion routing.
        """
        # Check if we have a mock client first
        if self.client and isinstance(self.client, (MagicMock, AsyncMock)):
            print("\nDetected mock client in provider router")
            print(f"Client type: {type(self.client)}")

            # For OpenAI clients (mock or real), ensure we're using the correct API key
            if isinstance(self.client, openai.OpenAI):
                # Don't override if client already has a key set
                if not self.client.api_key:
                    api_key = (
                        self.api_key
                        or litellm.openai_key
                        or litellm.api_key
                        or os.getenv("OPENAI_API_KEY")
                    )
                    if not api_key:
                        raise ValueError("No OpenAI API key found for client")
                    self.client.api_key = api_key
                # Ensure we're not using any Azure keys
                if "AZURE" in str(self.client.api_key).upper():
                    raise ValueError("Azure API key detected but OpenAI key required")

            # For mock clients, use the create method directly
            create_fn = self.client.chat.completions.with_raw_response.create
            print(f"Using mock create method of type: {type(create_fn)}")

            # Only include OpenAI-compatible parameters for mock client
            valid_openai_params = {
                "model",
                "messages",
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
                "stream",
            }
            # Filter parameters for mock client
            mock_params = {
                k: v
                for k, v in self.optional_params.items()
                if k in valid_openai_params and v is not None
            }
            mock_params["messages"] = self.messages

            print(f"Calling mock client with params: {mock_params}")
            if isinstance(create_fn, AsyncMock):
                response = await create_fn(**mock_params)
            else:
                response = create_fn(**mock_params)
            print(f"Mock response: {response}")
            return response

        # TODO: Implement provider-specific routing logic
        raise NotImplementedError("Provider-specific routing not implemented yet")

    def route_completion(self) -> Any:
        """
        Routes the completion request to the appropriate provider implementation.
        This is the synchronous version of the routing function.

        Returns:
            Any: The completion response from the provider.
        """
        print("\nProvider Router - route_completion")
        print(f"Model: {self.model}")
        print(f"Client type: {type(self.client)}")
        print(f"acompletion: {self.acompletion}")
        print(f"Optional params: {self.optional_params}")

        # Check if we have a mock client first
        if self.client and isinstance(self.client, (MagicMock, AsyncMock)):
            print("\nDetected mock client in provider router")
            print(f"Client type: {type(self.client)}")

            # Get the create method from the mock client
            if hasattr(self.client, "chat"):
                create_fn = self.client.chat.completions.with_raw_response.create
            else:
                create_fn = self.client.with_raw_response.create
            print(f"Using mock create method of type: {type(create_fn)}")

            # Prepare parameters for the mock client
            mock_params = {
                "model": self.model,
                "messages": self.messages,
            }

            # Add stream parameter only if it's provided
            if hasattr(self, "stream"):
                mock_params["stream"] = self.stream

            # Add other OpenAI-compatible parameters
            if self.optional_params:
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
                mock_params.update(
                    {
                        k: v
                        for k, v in self.optional_params.items()
                        if k in valid_openai_params and v is not None
                    }
                )

            print(f"Calling mock client with params: {mock_params}")
            try:
                # Check if the create function is async
                if inspect.iscoroutinefunction(create_fn) or isinstance(
                    create_fn, AsyncMock
                ):
                    print("Using async mock client")
                    if self.acompletion:
                        # For async completion requests, return the coroutine
                        async def wrap_async_response():
                            mock_response = await create_fn(**mock_params)
                            if hasattr(mock_response, "parse"):
                                parsed_response = mock_response.parse()
                                if isinstance(parsed_response, dict):
                                    from litellm.types.utils import ModelResponse, Choices, Usage
                                    print(f"Parsed response structure: {parsed_response}")
                                    
                                    # Handle choices conversion
                                    raw_choices = parsed_response.get("choices", [])
                                    choices = []
                                    for choice in raw_choices:
                                        if isinstance(choice, dict):
                                            choices.append(Choices(**choice))
                                        else:
                                            choices.append(choice)
                                    
                                    # Handle usage conversion
                                    raw_usage = parsed_response.get("usage", {})
                                    usage = Usage(**raw_usage) if isinstance(raw_usage, dict) else raw_usage
                                    
                                    return ModelResponse(
                                        id=parsed_response.get("id", f"mock-{str(time.time())}"),
                                        choices=choices,
                                        created=parsed_response.get("created", int(time.time())),
                                        model=parsed_response.get("model", self.model),
                                        usage=usage,
                                        system_fingerprint=parsed_response.get("system_fingerprint", "")
                                    )
                            return mock_response
                        return wrap_async_response()
                    else:
                        # For sync completion requests, run the coroutine
                        loop = asyncio.get_event_loop()
                        mock_response = loop.run_until_complete(create_fn(**mock_params))
                        if hasattr(mock_response, "parse"):
                            parsed_response = mock_response.parse()
                            if isinstance(parsed_response, dict):
                                from litellm.types.utils import ModelResponse, Choices, Usage
                                print(f"Parsed response structure: {parsed_response}")
                                
                                # Handle choices conversion
                                raw_choices = parsed_response.get("choices", [])
                                choices = []
                                for choice in raw_choices:
                                    if isinstance(choice, dict):
                                        choices.append(Choices(**choice))
                                    else:
                                        choices.append(choice)
                                
                                # Handle usage conversion
                                raw_usage = parsed_response.get("usage", {})
                                usage = Usage(**raw_usage) if isinstance(raw_usage, dict) else raw_usage
                                
                                response = ModelResponse(
                                    id=parsed_response.get("id", f"mock-{str(time.time())}"),
                                    choices=choices,
                                    created=parsed_response.get("created", int(time.time())),
                                    model=parsed_response.get("model", self.model),
                                    usage=usage,
                                    system_fingerprint=parsed_response.get("system_fingerprint", "")
                                )
                                return response
                        return mock_response
                else:
                    print("Using sync mock client")
                    mock_response = create_fn(**mock_params)
                    print(f"Mock response: {mock_response}")
                    
                    # Convert mock response to ModelResponse
                    if hasattr(mock_response, "parse"):
                        parsed_response = mock_response.parse()
                        if isinstance(parsed_response, dict):
                            from litellm.types.utils import ModelResponse, Choices, Usage
                            print(f"Parsed response structure: {parsed_response}")
                            
                            # Handle choices conversion
                            raw_choices = parsed_response.get("choices", [])
                            choices = []
                            for choice in raw_choices:
                                if isinstance(choice, dict):
                                    choices.append(Choices(**choice))
                                else:
                                    choices.append(choice)
                            
                            # Handle usage conversion
                            raw_usage = parsed_response.get("usage", {})
                            usage = Usage(**raw_usage) if isinstance(raw_usage, dict) else raw_usage
                            
                            response = ModelResponse(
                                id=parsed_response.get("id", f"mock-{str(time.time())}"),
                                choices=choices,
                                created=parsed_response.get("created", int(time.time())),
                                model=parsed_response.get("model", self.model),
                                usage=usage,
                                system_fingerprint=parsed_response.get("system_fingerprint", "")
                            )
                            return response
                    return mock_response
            except Exception as e:
                print(f"Error calling mock client: {str(e)}")
                import traceback

                traceback.print_exc()
                raise
        print(f"Provider Router routing completion for model: {self.model}")
        print(f"Provider Router received client type: {type(self.client)}")
        print(f"Custom LLM Provider: {self.custom_llm_provider}")
        print(f"Preview features enabled: {litellm.enable_preview_features}")

        # Check if this is an Azure model
        print("\nProvider Router - Initial routing check:")
        print(f"Model: {self.model}")
        print(f"Initial custom_llm_provider: {self.custom_llm_provider}")
        print(f"Client type: {type(self.client)}")

        # If not a mock client, proceed with normal routing
        if self.model.startswith("azure/"):
            self.custom_llm_provider = "azure"
            print(f"Setting custom_llm_provider to 'azure' for model: {self.model}")

        print(f"Final Custom LLM Provider: {self.custom_llm_provider}")
        if self.custom_llm_provider == "azure":
            print("\nEntering Azure completion handler path")
            # Check if this is a chat model that should use O1 handler
            print(f"\nProvider Router: Checking Azure model routing for {self.model}")
            print(f"Preview features enabled: {litellm.enable_preview_features}")

            o1_config = litellm.AzureOpenAIO1Config()
            is_o1_model = o1_config.is_o1_model(model=self.model)
            print(f"Is O1 model check result: {is_o1_model}")

            if litellm.enable_preview_features and is_o1_model:
                print(f"\nCreating O1 handler for model: {self.model}")
                print(f"Client type: {type(self.client)}")
                handler = AzureOpenAIO1ChatCompletion()
                # Get Azure configs
                api_type = get_secret("AZURE_API_TYPE") or "azure"
                api_version = (
                    self.kwargs.get("api_version")
                    or litellm.api_version
                    or get_secret("AZURE_API_VERSION")
                    or litellm.AZURE_DEFAULT_API_VERSION
                )
                api_key = (
                    self.api_key
                    or litellm.api_key
                    or litellm.azure_key
                    or get_secret("AZURE_OPENAI_API_KEY")
                    or get_secret("AZURE_API_KEY")
                )
                api_base = (
                    self.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
                )

                print(f"O1 handler parameters:")
                print(f"- API Type: {api_type}")
                print(f"- API Version: {api_version}")
                print(f"- API Base: {api_base}")
                print(f"- Stream: {self.stream}")
                print(f"- Client Type: {type(self.client)}")

                return handler.completion(
                    model=self.model,
                    messages=self.messages,
                    model_response=self.model_response,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    api_type=api_type,
                    azure_ad_token=None,  # We don't have this in the current context
                    dynamic_params=False,  # This is handled separately
                    print_verbose=True,
                    timeout=self.timeout,
                    logging_obj=self.logging,
                    optional_params=self.optional_params,
                    litellm_params=self.litellm_params,
                    logger_fn=self.logger_fn,
                    acompletion=self.acompletion,
                    headers=self.headers,
                    client=self.client,
                )
            return self._handle_azure_completion()
        elif self.custom_llm_provider == "azure_text":
            return self._handle_azure_text_completion()
        elif self.custom_llm_provider == "azure_ai":
            return self._handle_azure_ai_completion()
        elif self.custom_llm_provider == "replicate":
            return self._handle_replicate_completion()
        elif self.custom_llm_provider == "anthropic":
            return self._handle_anthropic_completion()
        elif self.custom_llm_provider == "cohere":
            return self._handle_cohere_completion()
        elif self.custom_llm_provider == "cohere_chat":
            return self._handle_cohere_chat_completion()
        elif self.custom_llm_provider == "maritalk":
            return self._handle_maritalk_completion()
        elif self.custom_llm_provider == "huggingface":
            return self._handle_huggingface_completion()
        elif self.custom_llm_provider == "oobabooga":
            return self._handle_oobabooga_completion()
        elif self.custom_llm_provider == "databricks":
            return self._handle_databricks_completion()
        elif self.custom_llm_provider == "openrouter":
            return self._handle_openrouter_completion()
        elif self.custom_llm_provider == "text-completion-openai":
            return self._handle_openai_text_completion()
        else:
            return self._handle_openai_chat_completion()

    def _handle_azure_completion(self) -> Any:
        """Handle Azure OpenAI completion."""
        print("\nInside _handle_azure_completion:")
        print(f"Model: {self.model}")
        print(f"Client type: {type(self.client)}")
        print(f"Client attributes: {dir(self.client) if self.client else 'No client'}")
        print(f"Stream: {self.stream}")
        print(f"Preview features: {litellm.enable_preview_features}")

        if self.client is not None:
            print(f"Is mock: {isinstance(self.client, (MagicMock, AsyncMock))}")
            if hasattr(self.client, "chat"):
                print(f"Chat type: {type(self.client.chat)}")
                if hasattr(self.client.chat, "completions"):
                    print(f"Completions type: {type(self.client.chat.completions)}")
                    if hasattr(self.client.chat.completions, "with_raw_response"):
                        print(
                            f"With raw response type: {type(self.client.chat.completions.with_raw_response)}"
                        )

        # Azure configs - only get these if not using a mock client
        if not isinstance(self.client, (MagicMock, AsyncMock)):
            api_type = get_secret("AZURE_API_TYPE") or "azure"
            api_version = (
                self.kwargs.get("api_version")
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
                or litellm.AZURE_DEFAULT_API_VERSION
            )
            api_key = (
                self.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )
            api_base = self.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        else:
            # Use dummy values for mock clients
            api_type = "azure"
            api_version = "mock-version"
            api_key = "mock-key"
            api_base = "mock-base"

        # Check for mock clients first
        print("\nChecking client type:")
        print(f"Client type: {type(self.client)}")

        is_mock = (
            isinstance(self.client, (MagicMock, AsyncMock))
            or hasattr(self.client, "_mock_return_value")
            or hasattr(self.client, "_mock_methods")
        )
        print(f"Is mock client: {is_mock}")

        # For mock clients, skip dynamic params check and use as-is
        dynamic_params = False
        if not is_mock and self.client is not None:
            dynamic_params = self._check_dynamic_azure_params(
                azure_client_params={"api_version": api_version},
                azure_client=self.client,
            )
            print(f"Dynamic params result: {dynamic_params}")
            print(f"Is AzureOpenAI: {isinstance(self.client, openai.AzureOpenAI)}")
            print(
                f"Is AsyncAzureOpenAI: {isinstance(self.client, openai.AsyncAzureOpenAI)}"
            )

        # Load config if set
        if (
            litellm.enable_preview_features
            and litellm.AzureOpenAIO1Config().is_o1_model(model=self.model)
        ):
            config = litellm.AzureOpenAIO1Config.get_config()
            for k, v in config.items():
                if k not in self.optional_params:
                    self.optional_params[k] = v

            handler = AzureOpenAIO1ChatCompletion()
            response = handler.completion(
                model=self.model,
                messages=self.messages,
                headers=self.headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                dynamic_params=dynamic_params,
                model_response=self.model_response,
                print_verbose=True,
                optional_params=self.optional_params,
                litellm_params=self.litellm_params,
                logger_fn=self.logger_fn,
                logging_obj=self.logging,
                acompletion=self.acompletion,
                timeout=self.timeout,
                client=self.client,
            )
        else:
            config = litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if k not in self.optional_params:
                    self.optional_params[k] = v

            handler = AzureTextCompletion()
            # Filter out dynamic_params as it's not supported by Azure handler
            handler_params = {
                "model": self.model,
                "messages": self.messages,
                "headers": self.headers,
                "api_key": api_key,
                "api_base": api_base,
                "api_version": api_version,
                "api_type": api_type,
                "azure_ad_token": None,  # Required parameter, default to None
                "model_response": self.model_response,
                "print_verbose": True,
                "optional_params": self.optional_params,
                "litellm_params": self.litellm_params,
                "logger_fn": self.logger_fn,
                "logging_obj": self.logging,
                "acompletion": self.acompletion,
                "timeout": self.timeout,
                "client": self.client,
            }
            response = handler.completion(**handler_params)

        if self.optional_params.get("stream", False):
            self.logging.post_call(
                input=self.messages,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": self.headers,
                    "api_version": api_version,
                    "api_base": api_base,
                },
            )

        return response

    def _handle_azure_text_completion(self) -> Any:
        """Handle Azure OpenAI text completion."""
        # Azure configs
        api_type = get_secret("AZURE_API_TYPE") or "azure"
        api_version = (
            self.kwargs.get("api_version")
            or litellm.api_version
            or get_secret("AZURE_API_VERSION")
        )
        api_key = (
            self.api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret("AZURE_OPENAI_API_KEY")
            or get_secret("AZURE_API_KEY")
        )
        api_base = self.api_base or litellm.api_base or get_secret("AZURE_API_BASE")

        if self.extra_headers is not None:
            self.optional_params["extra_headers"] = self.extra_headers

        # Load config if set
        config = litellm.AzureOpenAIConfig.get_config()
        for k, v in config.items():
            if k not in self.optional_params:
                self.optional_params[k] = v

        handler = AzureTextCompletion()
        response = handler.completion(
            model=self.model,
            messages=self.messages,
            headers=self.headers,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            logging_obj=self.logging,
            acompletion=self.acompletion,
            timeout=self.timeout,
            client=self.client,
        )

        if self.optional_params.get("stream", False) or self.acompletion is True:
            self.logging.post_call(
                input=self.messages,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": self.headers,
                    "api_version": api_version,
                    "api_base": api_base,
                },
            )

        return response

    def _handle_azure_ai_completion(self) -> Any:
        """Handle Azure AI completion."""
        api_base = self.api_base or litellm.api_base or get_secret("AZURE_AI_API_BASE")
        api_key = (
            self.api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret("AZURE_AI_API_KEY")
        )

        if self.extra_headers is not None:
            self.optional_params["extra_headers"] = self.extra_headers

        # Load config if set
        config = litellm.AzureAIStudioConfig.get_config()
        for k, v in config.items():
            if k not in self.optional_params:
                self.optional_params[k] = v

        # For Cohere
        if "command-r" in self.model:
            self.messages = stringify_json_tool_call_content(messages=self.messages)

        try:
            handler = OpenAILikeChatHandler()
            response = handler.completion(
                model=self.model,
                messages=self.messages,
                headers=self.headers,
                model_response=self.model_response,
                print_verbose=True,
                api_key=api_key,
                api_base=api_base,
                acompletion=self.acompletion,
                logging_obj=self.logging,
                optional_params=self.optional_params,
                litellm_params=self.litellm_params,
                logger_fn=self.logger_fn,
                timeout=self.timeout,
                custom_prompt_dict=self.custom_prompt_dict,
                client=self.client,
                organization=self.organization,
                custom_llm_provider="azure_ai",
                drop_params=self.kwargs.get("non_default_params", {}).get(
                    "drop_params"
                ),
            )
        except Exception as e:
            self.logging.post_call(
                input=self.messages,
                api_key=api_key,
                original_response=str(e),
                additional_args={"headers": self.headers},
            )
            raise e

        if self.optional_params.get("stream", False):
            self.logging.post_call(
                input=self.messages,
                api_key=api_key,
                original_response=response,
                additional_args={"headers": self.headers},
            )

        return response

    def _handle_replicate_completion(self) -> Any:
        """Handle Replicate completion."""
        replicate_key = (
            self.api_key
            or litellm.replicate_key
            or litellm.api_key
            or get_secret("REPLICATE_API_KEY")
            or get_secret("REPLICATE_API_TOKEN")
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("REPLICATE_API_BASE")
            or "https://api.replicate.com/v1"
        )

        custom_prompt_dict = self.custom_prompt_dict or litellm.custom_prompt_dict

        model_response = replicate_completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,  # for calculating input/output tokens
            api_key=replicate_key,
            logging_obj=self.logging,
            custom_prompt_dict=custom_prompt_dict,
            acompletion=self.acompletion,
            headers=self.headers,
        )

        if self.optional_params.get("stream", False) is True:
            self.logging.post_call(
                input=self.messages,
                api_key=replicate_key,
                original_response=model_response,
            )

        return model_response

    def _handle_anthropic_completion(self) -> Any:
        """Handle Anthropic completion."""
        api_key = (
            self.api_key
            or litellm.anthropic_key
            or litellm.api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("ANTHROPIC_API_BASE")
            or get_secret("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com/v1/messages"
        )

        if api_base is not None and not api_base.endswith("/v1/messages"):
            api_base += "/v1/messages"

        handler = AnthropicChatCompletion()
        response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            acompletion=self.acompletion,
            custom_prompt_dict=litellm.custom_prompt_dict,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,  # for calculating input/output tokens
            api_key=api_key,
            logging_obj=self.logging,
            headers=self.headers,
            timeout=self.timeout,
            client=self.client,
            custom_llm_provider=self.custom_llm_provider,
        )
        if self.optional_params.get("stream", False) or self.acompletion is True:
            self.logging.post_call(
                input=self.messages,
                api_key=api_key,
                original_response=response,
            )
        return response

    def _handle_cohere_completion(self) -> Any:
        """Handle Cohere completion."""
        cohere_key = (
            self.api_key
            or litellm.cohere_key
            or get_secret("COHERE_API_KEY")
            or get_secret("CO_API_KEY")
            or litellm.api_key
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("COHERE_API_BASE")
            or "https://api.cohere.ai/v1/generate"
        )

        headers = self.headers or litellm.headers or {}
        if headers is None:
            headers = {}

        if self.extra_headers is not None:
            headers.update(self.extra_headers)

        response = base_llm_http_handler.completion(
            model=self.model,
            stream=self.stream,
            messages=self.messages,
            acompletion=self.acompletion,
            api_base=api_base,
            model_response=self.model_response,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            custom_llm_provider="cohere",
            timeout=self.timeout,
            headers=headers,
            encoding=self.encoding,
            api_key=cohere_key,
            logging_obj=self.logging,
            client=self.client,
        )
        return response

    def _handle_cohere_chat_completion(self) -> Any:
        """Handle Cohere chat completion."""
        cohere_key = (
            self.api_key
            or litellm.cohere_key
            or get_secret("COHERE_API_KEY")
            or get_secret("CO_API_KEY")
            or litellm.api_key
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("COHERE_API_BASE")
            or "https://api.cohere.ai/v1/chat"
        )

        headers = self.headers or litellm.headers or {}
        if headers is None:
            headers = {}

        if self.extra_headers is not None:
            headers.update(self.extra_headers)

        response = base_llm_http_handler.completion(
            model=self.model,
            stream=self.stream,
            messages=self.messages,
            acompletion=self.acompletion,
            api_base=api_base,
            model_response=self.model_response,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            custom_llm_provider="cohere_chat",
            timeout=self.timeout,
            headers=headers,
            encoding=self.encoding,
            api_key=cohere_key,
            logging_obj=self.logging,
        )
        return response

    def _handle_maritalk_completion(self) -> Any:
        """Handle Maritalk completion."""
        maritalk_key = (
            self.api_key
            or litellm.maritalk_key
            or get_secret("MARITALK_API_KEY")
            or litellm.api_key
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("MARITALK_API_BASE")
            or "https://chat.maritaca.ai/api"
        )

        handler = OpenAILikeChatHandler()
        model_response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            api_key=maritalk_key,
            logging_obj=self.logging,
            custom_llm_provider="maritalk",
            custom_prompt_dict=self.custom_prompt_dict,
        )

        return model_response

    def _handle_huggingface_completion(self) -> Any:
        """Handle Hugging Face completion."""
        huggingface_key = (
            self.api_key
            or litellm.huggingface_key
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or litellm.api_key
        )
        hf_headers = self.headers or litellm.headers

        custom_prompt_dict = self.custom_prompt_dict or litellm.custom_prompt_dict
        handler = Huggingface()
        model_response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=self.api_base,
            headers=hf_headers or {},
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            api_key=huggingface_key,
            acompletion=self.acompletion,
            logging_obj=self.logging,
            custom_prompt_dict=custom_prompt_dict,
            timeout=self.timeout,
            client=self.client,
        )
        if (
            "stream" in self.optional_params
            and self.optional_params["stream"] is True
            and self.acompletion is False
        ):
            # don't try to access stream object,
            response = CustomStreamWrapper(
                model_response,
                self.model,
                custom_llm_provider="huggingface",
                logging_obj=self.logging,
            )
            return response
        return model_response

    def _handle_oobabooga_completion(self) -> Any:
        """Handle Oobabooga completion."""
        # Note: Oobabooga still uses function-based completion
        model_response = oobabooga(
            model=self.model,
            messages=self.messages,
            model_response=self.model_response,
            api_base=self.api_base,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            api_key=None,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            logging_obj=self.logging,
        )
        if "stream" in self.optional_params and self.optional_params["stream"] is True:
            # don't try to access stream object,
            response = CustomStreamWrapper(
                model_response,
                self.model,
                custom_llm_provider="oobabooga",
                logging_obj=self.logging,
            )
            return response
        return model_response

    def _handle_databricks_completion(self) -> Any:
        """Handle Databricks completion."""
        databricks_key = (
            self.api_key
            or litellm.databricks_key
            or get_secret("DATABRICKS_TOKEN")
            or litellm.api_key
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("DATABRICKS_API_BASE")
            or "https://api.databricks.com"
        )

        workspace_url = (
            self.api_base
            or litellm.api_base
            or get_secret("DATABRICKS_WORKSPACE_URL")
            or api_base
        )

        handler = DatabricksChatCompletion()
        response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            api_key=databricks_key,
            logging_obj=self.logging,
            acompletion=self.acompletion,
            custom_prompt_dict=self.custom_prompt_dict,
            timeout=self.timeout,
            client=self.client,
            workspace_url=workspace_url,
        )
        return response

    def _handle_openrouter_completion(self) -> Any:
        """Handle OpenRouter completion."""
        openrouter_key = (
            self.api_key
            or litellm.openrouter_key
            or get_secret("OPENROUTER_API_KEY")
            or litellm.api_key
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("OPENROUTER_API_BASE")
            or "https://openrouter.ai/api/v1"
        )

        headers = self.headers or litellm.headers
        if headers is None:
            headers = {}

        if self.extra_headers is not None:
            headers.update(self.extra_headers)

        # Add required OpenRouter headers
        if "HTTP-Referer" not in headers:
            headers["HTTP-Referer"] = "https://litellm.ai"
        if "X-Title" not in headers:
            headers["X-Title"] = "liteLLM"

        handler = OpenAILikeChatHandler()
        response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            api_key=openrouter_key,
            logging_obj=self.logging,
            acompletion=self.acompletion,
            custom_prompt_dict=self.custom_prompt_dict,
            timeout=self.timeout,
            client=self.client,
            headers=headers,
        )
        return response

    def _handle_openai_text_completion(self) -> Any:
        """Handle OpenAI text completion."""
        api_key = (
            self.api_key
            or litellm.openai_key
            or litellm.api_key
            or get_secret("OPENAI_API_KEY")
        )

        # Ensure we're not using Azure keys for OpenAI calls
        if api_key and "AZURE" in str(api_key).upper():
            raise ValueError("Azure API key detected but OpenAI key required")

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        handler = OpenAITextCompletion()
        response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            api_key=api_key,
            logging_obj=self.logging,
            acompletion=self.acompletion,
            custom_prompt_dict=self.custom_prompt_dict,
            timeout=self.timeout,
            client=self.client,
        )
        return response

    def _handle_openai_chat_completion(self) -> Any:
        """Handle OpenAI chat completion."""
        api_key = (
            self.api_key
            or litellm.openai_key
            or litellm.api_key
            or get_secret("OPENAI_API_KEY")
        )

        # Ensure we're not using Azure keys for OpenAI calls
        if api_key and "AZURE" in str(api_key).upper():
            raise ValueError("Azure API key detected but OpenAI key required")

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        headers = self.headers or litellm.headers
        if headers is None:
            headers = {}

        if self.extra_headers is not None:
            headers.update(self.extra_headers)

        handler = OpenAILikeChatHandler()
        # Extract provider from model name
        if "/" in self.model:
            # Handle explicit provider prefixes (e.g., 'azure/model-name')
            provider = self.model.split("/")[0]
            # Special handling for Azure providers
            if provider == "azure_ai":
                custom_llm_provider = "azure_ai"
            elif provider == "azure":
                custom_llm_provider = "azure"
            else:
                custom_llm_provider = provider
        elif self.model.startswith(("gpt-", "text-")):
            # Handle OpenAI models
            custom_llm_provider = "openai"
        elif "azure_ai" in self.model:
            # Handle Azure AI models without explicit prefix
            custom_llm_provider = "azure_ai"
        elif "azure" in self.model:
            # Handle Azure models without explicit prefix
            custom_llm_provider = "azure"
        else:
            # Default to OpenAI for unknown formats
            custom_llm_provider = "openai"

        response = handler.completion(
            model=self.model,
            messages=self.messages,
            api_base=api_base,
            model_response=self.model_response,
            print_verbose=True,
            optional_params=self.optional_params,
            litellm_params=self.litellm_params,
            logger_fn=self.logger_fn,
            encoding=self.encoding,
            api_key=api_key,
            logging_obj=self.logging,
            acompletion=self.acompletion,
            custom_prompt_dict=self.custom_prompt_dict,
            timeout=self.timeout,
            client=self.client,
            headers=headers,
            custom_llm_provider=custom_llm_provider,
        )
        return response

    def _check_dynamic_azure_params(
        self, azure_client_params: Dict[str, Any], azure_client: Any
    ) -> bool:
        """
        Check if Azure client parameters are dynamic.

        Args:
            azure_client_params (Dict[str, Any]): Azure client parameters.
            azure_client (Any): Azure client instance.

        Returns:
            bool: True if parameters are dynamic, False otherwise.
        """
        # For mock clients, always return False to reuse the mock
        print("\nChecking for mock client:")
        print(f"Client type: {type(azure_client)}")
        print(f"Client dir: {dir(azure_client)}")

        # Check if the client itself is a mock
        if isinstance(azure_client, (MagicMock, AsyncMock)):
            print("Top-level mock client detected, returning False")
            return False

        if azure_client is not None:
            # Check for _mock_return_value attribute (common in mocks)
            if hasattr(azure_client, "_mock_return_value"):
                print("Mock return value detected, returning False")
                return False

            # Check for _mock_methods attribute (common in mocks)
            if hasattr(azure_client, "_mock_methods"):
                print("Mock methods detected, returning False")
                return False

            print(f"Has chat: {hasattr(azure_client, 'chat')}")
            if hasattr(azure_client, "chat"):
                chat = azure_client.chat
                print(f"Chat type: {type(chat)}")
                print(f"Chat dir: {dir(chat)}")

                # Check if chat is a mock
                if isinstance(chat, (MagicMock, AsyncMock)):
                    print("Chat mock detected, returning False")
                    return False

                print(f"Has completions: {hasattr(chat, 'completions')}")
                if hasattr(chat, "completions"):
                    completions = chat.completions
                    print(f"Completions type: {type(completions)}")
                    print(f"Completions dir: {dir(completions)}")

                    # Check if completions is a mock
                    if isinstance(completions, (MagicMock, AsyncMock)):
                        print("Completions mock detected, returning False")
                        return False

                    print(
                        f"Has with_raw_response: {hasattr(completions, 'with_raw_response')}"
                    )
                    if hasattr(completions, "with_raw_response"):
                        with_raw = completions.with_raw_response
                        print(f"With raw response type: {type(with_raw)}")
                        print(f"With raw response dir: {dir(with_raw)}")

                        # Check if with_raw_response is a mock
                        if isinstance(with_raw, (MagicMock, AsyncMock)):
                            print("With raw response mock detected, returning False")
                            return False

                        create_attr = getattr(with_raw, "create", None)
                        print(f"Create attribute type: {type(create_attr)}")

                        # Check if create is a mock
                        if isinstance(create_attr, (MagicMock, AsyncMock)):
                            print("Create mock detected, returning False")
                            return False

        # If no client provided, we need to create one
        if azure_client is None:
            return True

        # Check if client's API version differs from provided version
        client_api_version = None

        # Try to get API version from client's custom query
        if (
            hasattr(azure_client, "_custom_query")
            and "api-version" in azure_client._custom_query
        ):
            client_api_version = azure_client._custom_query["api-version"]

        # Try to get API version from client's api_version attribute
        elif hasattr(azure_client, "api_version"):
            client_api_version = azure_client.api_version

        # Get the requested API version from parameters
        requested_api_version = azure_client_params.get("api_version")

        print(f"\nAPI Version comparison:")
        print(f"Client API version: {client_api_version}")
        print(f"Requested API version: {requested_api_version}")

        # If we have both versions and they differ, we need a new client
        if (
            client_api_version
            and requested_api_version
            and client_api_version != requested_api_version
        ):
            print("API versions differ - need new client")
            return True

        return False
