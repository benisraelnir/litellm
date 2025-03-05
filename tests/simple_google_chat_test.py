"""
Simple test script for Google Chat notifications.

This script tests the Google Chat notification functionality directly,
without relying on the entire LiteLLM package.
"""

import asyncio
import httpx
import sys
from typing import Any, Dict, Optional


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
    try:
        print(f"Sending notification to Google Chat: {message}")

        # Prepare the message payload
        payload: Dict[str, Any] = {"text": message}

        # Add thread key if provided
        if thread_key:
            thread_data: Dict[str, str] = {"threadKey": thread_key}
            payload["thread"] = thread_data

        # Send the request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10,
            )

        # Check if the request was successful
        if response.status_code >= 400:
            error_msg = f"Failed to send Google Chat notification. Status code: {response.status_code}, Response: {response.text}"
            print(error_msg)
            return {
                "success": False,
                "status_code": response.status_code,
                "message": error_msg,
            }

        print("Successfully sent notification to Google Chat")
        return {
            "success": True,
            "status_code": response.status_code,
            "message": "Successfully sent notification to Google Chat",
        }

    except Exception as e:
        error_msg = f"Error sending Google Chat notification: {str(e)}"
        print(error_msg)
        return {"success": False, "message": error_msg}


async def run_tests():
    """Run tests for Google Chat notifications."""
    # Test 1: Error handling with invalid webhook URL
    print("\nTest 1: Testing error handling with an invalid webhook URL...")
    result1 = await send_chat_notification(
        webhook_url="https://invalid-webhook-url.example.com",
        message="This message should fail to send",
    )
    
    if not result1.get("success", True):
        print("✅ Test 1 passed: Correctly handled invalid webhook URL")
    else:
        print("❌ Test 1 failed: Did not properly handle invalid webhook URL")
    
    # Test 2: Local server test (if available)
    try:
        import socket
        
        # Check if port 8080 is available on localhost
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 8080))
            s.listen(1)
            
            # Start a simple HTTP server in a separate process
            import subprocess
            import time
            
            server_process = subprocess.Popen(
                [sys.executable, '-m', 'http.server', '8080'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(2)
            
            print("\nTest 2: Sending notification to local test server...")
            result2 = await send_chat_notification(
                webhook_url="http://localhost:8080",
                message="Test notification to local server",
            )
            
            # Stop the server
            server_process.terminate()
            
            # Check result
            print(f"Test 2 result: {result2}")
            
    except Exception as e:
        print(f"Skipping Test 2 (local server test): {str(e)}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(run_tests())
