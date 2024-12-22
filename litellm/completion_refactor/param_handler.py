from typing import Any, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel

import litellm
from litellm.types.llms.openai import (
    ChatCompletionAssistantMessage,
    ChatCompletionAudioParam,
    ChatCompletionModality,
    ChatCompletionPredictionContentParam,
    ChatCompletionUserMessage,
)
from litellm.utils import ModelResponse, get_secret


class CompletionParamsHandler:
    """
    Handles parameter processing and validation for the completion method.
    Extracts and validates parameters from the input kwargs according to the
    completion method's requirements.

    Args:
        **kwargs: Keyword arguments containing all possible completion parameters.
            See the completion method's docstring for detailed parameter descriptions.
    """

    def __init__(self, **kwargs):
        # Store raw kwargs
        self.kwargs = kwargs

        # Core parameters
        self.model: str = kwargs.get("model")
        self.messages: List[Dict[str, str]] = kwargs.get("messages", [])
        self.timeout: Optional[Union[float, str, httpx.Timeout]] = kwargs.get("timeout")
        self.temperature: Optional[float] = kwargs.get("temperature")
        self.api_key: Optional[str] = kwargs.get("api_key")
        self.api_base: Optional[str] = kwargs.get("base_url") or kwargs.get("api_base")
        self.custom_llm_provider: Optional[str] = kwargs.get("custom_llm_provider")

        # OpenAI-style parameters
        self.top_p: Optional[float] = kwargs.get("top_p")
        self.n: Optional[int] = kwargs.get("n")
        self.stream: Optional[bool] = kwargs.get("stream")
        self.stream_options: Optional[dict] = kwargs.get("stream_options")
        self.stop = kwargs.get("stop")
        self.max_tokens: Optional[int] = kwargs.get("max_tokens")
        self.max_completion_tokens: Optional[int] = kwargs.get("max_completion_tokens")
        self.presence_penalty: Optional[float] = kwargs.get("presence_penalty")
        self.frequency_penalty: Optional[float] = kwargs.get("frequency_penalty")
        self.logit_bias: Optional[dict] = kwargs.get("logit_bias")
        self.user: Optional[str] = kwargs.get("user")

        # OpenAI v1.0+ parameters
        self.response_format: Optional[Union[dict, Type[BaseModel]]] = kwargs.get(
            "response_format"
        )
        self.seed: Optional[int] = kwargs.get("seed")
        self.tools: Optional[List] = kwargs.get("tools")
        self.tool_choice: Optional[Union[str, dict]] = kwargs.get("tool_choice")
        self.logprobs: Optional[bool] = kwargs.get("logprobs")
        self.top_logprobs: Optional[int] = kwargs.get("top_logprobs")
        self.parallel_tool_calls: Optional[bool] = kwargs.get("parallel_tool_calls")

        # Modality and prediction parameters
        self.modalities: Optional[List[ChatCompletionModality]] = kwargs.get(
            "modalities"
        )
        self.prediction: Optional[ChatCompletionPredictionContentParam] = kwargs.get(
            "prediction"
        )
        self.audio: Optional[ChatCompletionAudioParam] = kwargs.get("audio")

        # Soon to be deprecated OpenAI params
        self.functions: Optional[List] = kwargs.get("functions")
        self.function_call: Optional[str] = kwargs.get("function_call")

        # API configuration
        self.api_version: Optional[str] = kwargs.get("api_version")
        self.model_list: Optional[list] = kwargs.get("model_list")
        self.deployment_id = kwargs.get("deployment_id")
        self.headers = kwargs.get("headers") or kwargs.get("extra_headers")

        # LiteLLM specific parameters
        self.model_response = kwargs.get("model_response", ModelResponse())
        self.litellm_logging_obj = kwargs.get(
            "litellm_logging_obj"
        )  # Initialize logging object
        self.logger_fn = kwargs.get("logger_fn")
        self.verbose = kwargs.get("verbose", False)
        self.custom_prompt_dict = kwargs.get("custom_prompt_dict")
        self.metadata = kwargs.get("metadata")
        self.model_info = kwargs.get("model_info")
        self.proxy_server_request = kwargs.get("proxy_server_request")
        self.fallbacks = kwargs.get("fallbacks")
        self.client = kwargs.get("client")  # Initialize client parameter

        # Completion mode flags
        self.acompletion = kwargs.get("acompletion", False)
        self.text_completion = kwargs.get("text_completion", False)
        self.atext_completion = kwargs.get("atext_completion", False)

        # Message continuation parameters
        self.ensure_alternating_roles: Optional[bool] = kwargs.get(
            "ensure_alternating_roles"
        )
        self.user_continue_message: Optional[ChatCompletionUserMessage] = kwargs.get(
            "user_continue_message"
        )
        self.assistant_continue_message: Optional[ChatCompletionAssistantMessage] = (
            kwargs.get("assistant_continue_message")
        )

        # Initialize processed parameters
        self.processed_params: Dict[str, Any] = {}

    def process_arguments(self) -> Dict[str, Any]:
        """
        Process and validate all input parameters.
        Returns a dictionary of processed parameters.

        Returns:
            Dict[str, Any]: Processed and validated parameters ready for the completion call.
        """
        self._validate_required_params()
        self._process_api_params()
        self._process_optional_params()
        self._register_custom_pricing()

        return self.processed_params

    def _validate_required_params(self):
        """
        Validate required parameters are present and in correct format.

        Raises:
            ValueError: If required parameters are missing or invalid.
            TypeError: If parameters are of incorrect type.
        """
        if not self.model:
            raise ValueError("model parameter must be provided")
        if not isinstance(self.messages, list):
            raise TypeError("messages must be a list")

        # Validate numeric parameters
        if self.temperature is not None and not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be a number")
        if self.max_tokens is not None and not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")

    def _process_api_params(self):
        """
        Process API-related parameters (keys, bases, etc).
        Sets up authentication and endpoint configuration.
        """
        # API Key processing
        self.processed_params["api_key"] = (
            self.api_key or litellm.api_key or get_secret("OPENAI_API_KEY")
        )

        # API Base processing
        self.processed_params["api_base"] = (
            self.api_base or litellm.api_base or get_secret("OPENAI_API_BASE")
        )

        # API Version
        if self.api_version:
            self.processed_params["api_version"] = self.api_version

    def _process_optional_params(self):
        """
        Process all optional parameters including OpenAI-specific and custom parameters.
        Handles both standard and provider-specific optional parameters.
        """
        optional_params = {}

        # OpenAI completion parameters
        openai_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
            "stream_options": self.stream_options,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "user": self.user,
            "response_format": self.response_format,
            "seed": self.seed,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "parallel_tool_calls": self.parallel_tool_calls,
        }

        # Add non-None OpenAI parameters
        for key, value in openai_params.items():
            if value is not None:
                optional_params[key] = value

        # Handle function calling parameters (soon to be deprecated)
        if self.functions is not None:
            optional_params["functions"] = self.functions
        if self.function_call is not None:
            optional_params["function_call"] = self.function_call

        # Handle modality and prediction parameters
        if self.modalities is not None:
            optional_params["modalities"] = self.modalities
        if self.prediction is not None:
            optional_params["prediction"] = self.prediction
        if self.audio is not None:
            optional_params["audio"] = self.audio

        # Add any remaining kwargs that weren't explicitly processed
        excluded_keys = set(
            [
                "model",
                "messages",
                "timeout",
                "api_key",
                "api_base",
                "custom_llm_provider",
                "model_response",
                "litellm_logging_obj",
                "acompletion",
                "stream",
                "headers",
                "custom_prompt_dict",
                "logger_fn",
                "verbose",
                "input_cost_per_token",
                "output_cost_per_token",
                "input_cost_per_second",
                "output_cost_per_second",
                "client",  # Exclude client to prevent duplicate
            ]
            + list(openai_params.keys())
        )

        for key, value in self.kwargs.items():
            if key not in excluded_keys and value is not None:
                optional_params[key] = value

        self.processed_params["optional_params"] = optional_params

        # Add core processed parameters
        self.processed_params.update(
            {
                "model": self.model,
                "client": self.client,  # Add client to processed params
                "messages": self.messages,
                "timeout": self.timeout,
                "custom_llm_provider": self.custom_llm_provider,
                "model_response": self.model_response,
                "logging_obj": self.litellm_logging_obj,
                "acompletion": self.acompletion,
                "headers": self.headers,
                "custom_prompt_dict": self.custom_prompt_dict,
                "logger_fn": self.logger_fn,
                "verbose": self.verbose,
                "metadata": self.metadata,
                "model_info": self.model_info,
                "proxy_server_request": self.proxy_server_request,
                "fallbacks": self.fallbacks,
                "client": self.client,  # Add client to processed params
            }
        )

    def _register_custom_pricing(self):
        """Register custom model pricing if provided."""
        if (
            self.kwargs.get("input_cost_per_token") is not None
            and self.kwargs.get("output_cost_per_token") is not None
        ):
            litellm.register_model(
                {
                    f"{self.custom_llm_provider}/{self.model}": {
                        "input_cost_per_token": self.kwargs["input_cost_per_token"],
                        "output_cost_per_token": self.kwargs["output_cost_per_token"],
                        "litellm_provider": self.custom_llm_provider,
                    }
                }
            )
        elif self.kwargs.get("input_cost_per_second") is not None:
            litellm.register_model(
                {
                    f"{self.custom_llm_provider}/{self.model}": {
                        "input_cost_per_second": self.kwargs["input_cost_per_second"],
                        "output_cost_per_second": self.kwargs.get(
                            "output_cost_per_second"
                        ),
                        "litellm_provider": self.custom_llm_provider,
                    }
                }
            )
