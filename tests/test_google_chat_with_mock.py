"""
Test script for Google Chat notifications using a mock server.

This script demonstrates how to use the Google Chat notification functionality
with a mock server to verify that the setup works correctly without needing
an actual Google Chat webhook URL.
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, Optional

# Add the parent directory to the path so we can import litellm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litellm.integrations.GoogleChatAlerting import send_chat_notification
from tests.mock_google_chat_server import MockGoogleChatServer


async def run_tests():
    """Run tests for Google Chat notifications using a mock server."""
    # Start the mock server
    print("Starting mock Google Chat server...")
    server = MockGoogleChatServer()
    await server.start()
    
    webhook_url = server.get_webhook_url()
    print(f"Mock server running at {webhook_url}")
    
    # Test 1: Basic notification
    print("\nTest 1: Sending basic notification...")
    result1 = await send_chat_notification(
        webhook_url=webhook_url,
        message="Test notification from LiteLLM",
    )
    
    if result1.get("success", False):
        print("✅ Test 1 passed: Basic notification sent successfully!")
    else:
        print(f"❌ Test 1 failed: {result1.get('message', 'Unknown error')}")
    
    # Wait a moment for the server to process the message
    await asyncio.sleep(1)
    
    # Verify the message was received by the mock server
    messages = server.get_received_messages()
    if len(messages) > 0 and messages[0].get("text") == "Test notification from LiteLLM":
        print("✅ Message content verification passed!")
    else:
        print("❌ Message content verification failed!")
    
    # Test 2: Notification with thread key
    print("\nTest 2: Sending notification with thread key...")
    result2 = await send_chat_notification(
        webhook_url=webhook_url,
        message="Test notification with thread key",
        thread_key="test-thread-123",
    )
    
    if result2.get("success", False):
        print("✅ Test 2 passed: Notification with thread key sent successfully!")
    else:
        print(f"❌ Test 2 failed: {result2.get('message', 'Unknown error')}")
    
    # Wait a moment for the server to process the message
    await asyncio.sleep(1)
    
    # Verify the thread key was included
    messages = server.get_received_messages()
    if (
        len(messages) > 1 
        and messages[1].get("text") == "Test notification with thread key"
        and messages[1].get("thread", {}).get("threadKey") == "test-thread-123"
    ):
        print("✅ Thread key verification passed!")
    else:
        print("❌ Thread key verification failed!")
    
    # Test 3: Error handling with invalid webhook URL
    print("\nTest 3: Testing error handling with an invalid webhook URL...")
    result3 = await send_chat_notification(
        webhook_url="https://invalid-webhook-url.example.com",
        message="This message should fail to send",
    )
    
    if not result3.get("success", True):
        print("✅ Test 3 passed: Correctly handled invalid webhook URL")
    else:
        print("❌ Test 3 failed: Did not properly handle invalid webhook URL")
    
    print("\nAll tests completed!")
    
    # Return a summary of the test results
    return {
        "test1_success": result1.get("success", False),
        "test2_success": result2.get("success", False),
        "test3_success": not result3.get("success", True),
        "messages_received": len(server.get_received_messages()),
    }


if __name__ == "__main__":
    # Run the tests
    test_results = asyncio.run(run_tests())
    
    # Print a summary of the test results
    print("\nTest Results Summary:")
    print(f"Basic notification test: {'Passed' if test_results['test1_success'] else 'Failed'}")
    print(f"Thread key test: {'Passed' if test_results['test2_success'] else 'Failed'}")
    print(f"Error handling test: {'Passed' if test_results['test3_success'] else 'Failed'}")
    print(f"Total messages received by mock server: {test_results['messages_received']}")
    
    # Exit with appropriate status code
    if all([test_results['test1_success'], test_results['test2_success'], test_results['test3_success']]):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
