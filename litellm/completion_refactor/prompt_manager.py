from typing import Dict, List, Optional, Union, cast
from litellm.utils import get_completion_messages
from litellm.types import (
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage
)

class PromptManager:
    """
    Manages prompt processing and message formatting for completion requests.
    Handles message structure, roles, and custom prompt templates.

    Args:
        messages (List): List of message dictionaries representing the conversation.
        custom_prompt_dict (Optional[Dict]): Custom prompt template configuration.
        ensure_alternating_roles (Optional[bool]): Whether to enforce alternating message roles.
        user_continue_message (Optional[ChatCompletionUserMessage]): Message to continue user context.
        assistant_continue_message (Optional[ChatCompletionAssistantMessage]): Message to continue assistant context.
    """
    def __init__(
        self,
        messages: List,
        custom_prompt_dict: Optional[Dict] = None,
        ensure_alternating_roles: Optional[bool] = None,
        user_continue_message: Optional[ChatCompletionUserMessage] = None,
        assistant_continue_message: Optional[ChatCompletionAssistantMessage] = None,
    ):
        self.messages = messages
        self.custom_prompt_dict = custom_prompt_dict or {}
        self.ensure_alternating_roles = ensure_alternating_roles or False
        self.user_continue_message = user_continue_message
        self.assistant_continue_message = assistant_continue_message
        
        # Initialize prompt template components
        self.initial_prompt_value = None
        self.roles = None
        self.final_prompt_value = None
        self.bos_token = None
        self.eos_token = None
        
        # Extract prompt template components if provided
        if custom_prompt_dict:
            for model, config in custom_prompt_dict.items():
                self.initial_prompt_value = config.get("initial_prompt_value")
                self.roles = config.get("roles")
                self.final_prompt_value = config.get("final_prompt_value")
                self.bos_token = config.get("bos_token")
                self.eos_token = config.get("eos_token")
                break  # Only use the first model's config for now
    
    def prepare_messages(self, model: str) -> List[Dict[str, str]]:
        """
        Prepare and format messages according to the configuration.
        
        Args:
            model (str): The model name to use for prompt template selection.
            
        Returns:
            List[Dict[str, str]]: Processed and formatted messages ready for completion.
        """
        # Get processed messages with role alternation if needed
        processed_messages = get_completion_messages(
            messages=self.messages,
            ensure_alternating_roles=self.ensure_alternating_roles,
            user_continue_message=self.user_continue_message,
            assistant_continue_message=self.assistant_continue_message,
        )
        
        # Apply custom prompt template if available for the model
        if model in self.custom_prompt_dict:
            processed_messages = self._apply_prompt_template(
                processed_messages,
                self.custom_prompt_dict[model]
            )
        
        return processed_messages
    
    def _apply_prompt_template(
        self,
        messages: List[Dict[str, str]],
        template_config: Dict
    ) -> List[Dict[str, str]]:
        """
        Apply custom prompt template to the messages.
        
        Args:
            messages (List[Dict[str, str]]): Original messages to format.
            template_config (Dict): Template configuration for the specific model.
            
        Returns:
            List[Dict[str, str]]: Messages formatted according to the template.
        """
        formatted_messages = []
        
        # Add initial prompt if specified
        if template_config.get("initial_prompt_value"):
            formatted_messages.append({
                "role": "system",
                "content": template_config["initial_prompt_value"]
            })
        
        # Process each message with custom roles if specified
        roles = template_config.get("roles", {})
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Apply role mapping if available
            if roles and role in roles:
                role = roles[role]
            
            # Apply tokens if specified
            if template_config.get("bos_token"):
                content = template_config["bos_token"] + content
            if template_config.get("eos_token"):
                content = content + template_config["eos_token"]
            
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        # Add final prompt if specified
        if template_config.get("final_prompt_value"):
            formatted_messages.append({
                "role": "system",
                "content": template_config["final_prompt_value"]
            })
        
        return formatted_messages
    
    def validate_messages(self) -> None:
        """
        Validate message format and structure.
        
        Raises:
            ValueError: If messages are invalid or missing required fields.
            TypeError: If message format is incorrect.
        """
        if not isinstance(self.messages, list):
            raise TypeError("Messages must be a list")
        
        for message in self.messages:
            if not isinstance(message, dict):
                raise TypeError("Each message must be a dictionary")
            if "role" not in message:
                raise ValueError("Each message must have a 'role' field")
            if "content" not in message:
                raise ValueError("Each message must have a 'content' field")
            if not isinstance(message["role"], str):
                raise TypeError("Message role must be a string")
            if not isinstance(message["content"], str):
                raise TypeError("Message content must be a string")
