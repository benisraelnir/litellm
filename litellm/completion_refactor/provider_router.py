from typing import Dict, List, Optional, Union, Any
import os
import openai
from litellm.utils import get_secret, get_secret_str
import litellm
from litellm.llms import (
    azure_chat_completions,
    azure_text_completions,
    azure_o1_chat_completions,
    openai_chat_completions,
    openai_text_completions,
    huggingface,
    anthropic_chat_completions,
    base_llm_http_handler,
    oobabooga,
    databricks_chat_completions,
    replicate_chat_completion,
    openai_like_chat_completion,
)
from litellm.utils import CustomStreamWrapper, stringify_json_tool_call_content

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
        **kwargs
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
        
    def route_completion(self) -> Any:
        """
        Routes the completion request to the appropriate provider implementation.
        
        Returns:
            Any: The completion response from the provider.
        """
        if self.custom_llm_provider == "azure":
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
        # Azure configs
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
        
        
        # Check dynamic params
        dynamic_params = False
        if self.client is not None and (
            isinstance(self.client, openai.AzureOpenAI)
            or isinstance(self.client, openai.AsyncAzureOpenAI)
        ):
            dynamic_params = self._check_dynamic_azure_params(
                azure_client_params={"api_version": api_version},
                azure_client=self.client,
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

            response = azure_o1_chat_completions.completion(
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

            response = azure_chat_completions.completion(
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

        response = azure_text_completions.completion(
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
        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("AZURE_AI_API_BASE")
        )
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
            response = openai_chat_completions.completion(
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
                drop_params=self.kwargs.get("non_default_params", {}).get("drop_params"),
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

        model_response = replicate_chat_completion(  # type: ignore
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
        custom_prompt_dict = self.custom_prompt_dict or litellm.custom_prompt_dict
        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("ANTHROPIC_API_BASE")
            or get_secret("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com/v1/messages"
        )

        if api_base is not None and not api_base.endswith("/v1/messages"):
            api_base += "/v1/messages"

        response = anthropic_chat_completions.completion(
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
            or get_secret_str("COHERE_API_KEY")
            or get_secret_str("CO_API_KEY")
            or litellm.api_key
        )

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret_str("COHERE_API_BASE")
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

        model_response = openai_like_chat_completion.completion(
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
        custom_llm_provider = "huggingface"
        huggingface_key = (
            self.api_key
            or litellm.huggingface_key
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or litellm.api_key
        )
        hf_headers = self.headers or litellm.headers

        custom_prompt_dict = self.custom_prompt_dict or litellm.custom_prompt_dict
        model_response = huggingface.completion(
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
        custom_llm_provider = "oobabooga"
        model_response = oobabooga.completion(
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

        response = databricks_chat_completions.completion(
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

        response = openai_chat_completions.completion(
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

        api_base = (
            self.api_base
            or litellm.api_base
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        response = openai_text_completions.completion(
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

        response = openai_chat_completions.completion(
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
        # Check if client's API version differs from provided version
        if hasattr(azure_client, "api_version"):
            client_api_version = getattr(azure_client, "api_version")
            if client_api_version != azure_client_params.get("api_version"):
                return True
        
        # Check if client's base URL differs from provided base URL
        if hasattr(azure_client, "base_url"):
            client_base_url = getattr(azure_client, "base_url")
            if client_base_url != self.api_base:
                return True
        
        # Check if client's deployment ID differs from model name
        if hasattr(azure_client, "deployment"):
            client_deployment = getattr(azure_client, "deployment")
            if client_deployment != self.model:
                return True
        
        return False
