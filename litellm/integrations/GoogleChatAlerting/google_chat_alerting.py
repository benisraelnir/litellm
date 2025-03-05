"""
Functions for sending Google Chat notifications.

This module provides functionality to send notifications to Google Chat spaces
using webhooks, suitable for automated pipeline use.
"""

import logging
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.proxy._types import WebhookEvent

# Set up logging
logger = logging.getLogger(__name__)


class GoogleChatNotifier:
    """A class to handle sending notifications to Google Chat."""

    def __init__(self, webhook_url: str, timeout: int = 10):
        """
        Initialize the Google Chat notifier.

        Args:
            webhook_url: The webhook URL for the Google Chat space.
            timeout: Request timeout in seconds.
        """
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def send_notification(
        self, message: str, thread_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a notification to Google Chat.

        Args:
            message: The message to send.
            thread_key: Optional thread key to post in a specific thread.

        Returns:
            Dict containing response information.

        Raises:
            Exception: If the notification fails to send.
        """
        try:
            verbose_logger.debug(
                "GoogleChatAlerting: Sending notification to Google Chat"
            )

            # Prepare the message payload
            payload: Dict[str, Any] = {"text": message}

            # Add thread key if provided
            if thread_key:
                thread_data: Dict[str, str] = {"threadKey": thread_key}
                payload["thread"] = thread_data

            # Send the request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                )

            # Check if the request was successful
            if response.status_code >= 400:
                error_msg = f"Failed to send Google Chat notification. Status code: {response.status_code}, Response: {response.text}"
                verbose_logger.error(error_msg)
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "message": error_msg,
                }

            verbose_logger.debug(
                "GoogleChatAlerting: Successfully sent notification to Google Chat"
            )
            return {
                "success": True,
                "status_code": response.status_code,
                "message": "Successfully sent notification to Google Chat",
            }

        except Exception as e:
            error_msg = f"Error sending Google Chat notification: {str(e)}"
            verbose_logger.error(error_msg)
            return {"success": False, "message": error_msg}


async def send_chat_notification(
    webhook_url: str, message: str, thread_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send a notification to Google Chat.

    Args:
        webhook_url: The webhook URL for the Google Chat space.
        message: The message to send.
        thread_key: Optional thread key to post in a specific thread.

    Returns:
        Dict containing response information.
    """
    notifier = GoogleChatNotifier(webhook_url=webhook_url)
    return await notifier.send_notification(message=message, thread_key=thread_key)


async def send_team_budget_alert(webhook_event: WebhookEvent, webhook_url: str) -> bool:
    """
    Send a Google Chat Alert when the Team Budget is crossed.

    Args:
        webhook_event: The webhook event containing budget information.
        webhook_url: The webhook URL for the Google Chat space.

    Returns:
        True if sent successfully, False otherwise.
    """
    team_alias = webhook_event.team_alias
    verbose_logger.debug(
        "GoogleChatAlerting: Sending Team Budget Alert for team=%s", team_alias
    )

    max_budget = webhook_event.max_budget
    current_spend = webhook_event.spend

    message = (
        f"🚨 *Budget Alert for Team {team_alias}* 🚨\n\n"
        f"Your team's LLM API usage has crossed its budget of ${max_budget}, "
        f"current spend is ${current_spend}.\n\n"
        f"API requests will be rejected until either:\n"
        f"- You increase your budget, or\n"
        f"- Your budget gets reset\n\n"
        f"This alert was sent via LiteLLM."
    )

    result = await send_chat_notification(webhook_url=webhook_url, message=message)
    return result.get("success", False)
