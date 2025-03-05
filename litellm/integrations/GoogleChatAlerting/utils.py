"""
Utility functions for Google Chat alerting.
"""

import os
from typing import Any, Dict, Optional


def get_webhook_url_from_env() -> Optional[str]:
    """
    Get the Google Chat webhook URL from environment variables.

    Returns:
        The webhook URL if set, None otherwise.
    """
    return os.getenv("GOOGLE_CHAT_WEBHOOK_URL")


def format_message(
    title: str, message: str, additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a message for Google Chat.

    Args:
        title: The title of the message.
        message: The main message content.
        additional_info: Optional dictionary of additional information to include.

    Returns:
        Formatted message string.
    """
    formatted_message = f"*{title}*\n\n{message}"

    if additional_info:
        formatted_message += "\n\n*Additional Information:*"
        for key, value in additional_info.items():
            formatted_message += f"\n• {key}: {value}"

    return formatted_message
