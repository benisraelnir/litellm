"""
Test script for Google Chat notifications.

This script demonstrates how to use the Google Chat notification functionality
and verifies that the setup works correctly.
"""

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, Optional

# Add the parent directory to the path so we can import litellm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litellm.integrations.GoogleChatAlerting import (
    get_webhook_url_from_env,
    send_chat_notification,
)


async def test_notification(
    webhook_url: Optional[str] = None,
    message: str = "Test notification from LiteLLM",
    thread_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Test sending a notification to Google Chat.

    Args:
        webhook_url: The webhook URL for the Google Chat space.
            If not provided, will try to get it from environment variables.
        message: The message to send.
        thread_key: Optional thread key to post in a specific thread.

    Returns:
        Dict containing response information.
    """
    # If webhook_url is not provided, try to get it from environment variables
    if not webhook_url:
        webhook_url = get_webhook_url_from_env()
        if not webhook_url:
            print(
                "Error: No webhook URL provided and GOOGLE_CHAT_WEBHOOK_URL environment variable not set."
            )
            print(
                "Please provide a webhook URL using the --webhook-url argument or set the GOOGLE_CHAT_WEBHOOK_URL environment variable."
            )
            return {"success": False, "message": "No webhook URL provided"}

    print("Sending test notification to Google Chat...")
    result = await send_chat_notification(
        webhook_url=webhook_url, message=message, thread_key=thread_key
    )

    if result.get("success", False):
        print("✅ Notification sent successfully!")
    else:
        print(
            f"❌ Failed to send notification: {result.get('message', 'Unknown error')}"
        )

    return result


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Google Chat notifications for LiteLLM"
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        help="The webhook URL for the Google Chat space. If not provided, will try to get it from GOOGLE_CHAT_WEBHOOK_URL environment variable.",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Test notification from LiteLLM",
        help="The message to send to Google Chat.",
    )
    parser.add_argument(
        "--thread-key",
        type=str,
        help="Optional thread key to post in a specific thread.",
    )
    return parser.parse_args()


async def main():
    """Main function to run the test."""
    args = parse_arguments()

    # Test basic notification
    result = await test_notification(
        webhook_url=args.webhook_url,
        message=args.message,
        thread_key=args.thread_key,
    )

    # Test error handling with an invalid webhook URL
    if result.get("success", False):
        print("\nTesting error handling with an invalid webhook URL...")
        error_result = await test_notification(
            webhook_url="https://invalid-webhook-url.example.com",
            message="This message should fail to send",
        )
        if not error_result.get("success", True):
            print(
                "✅ Error handling test passed: Correctly handled invalid webhook URL"
            )
        else:
            print(
                "❌ Error handling test failed: Did not properly handle invalid webhook URL"
            )


if __name__ == "__main__":
    asyncio.run(main())
