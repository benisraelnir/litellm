"""
Mock Google Chat webhook server for testing Google Chat notifications.

This script creates a simple HTTP server that simulates a Google Chat webhook endpoint.
It's useful for testing the Google Chat notification functionality without needing
an actual Google Chat webhook URL.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional

from aiohttp import web

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockGoogleChatServer:
    """A mock server that simulates a Google Chat webhook endpoint."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Initialize the mock server.

        Args:
            host: The host to bind the server to.
            port: The port to bind the server to.
        """
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_post("/", self.handle_webhook)
        self.received_messages: list = []

    async def handle_webhook(self, request: web.Request) -> web.Response:
        """
        Handle incoming webhook requests.

        Args:
            request: The incoming request.

        Returns:
            A response to send back to the client.
        """
        try:
            # Parse the request body
            body = await request.json()
            logger.info(f"Received webhook request: {body}")

            # Store the received message
            self.received_messages.append(body)

            # Return a success response
            return web.json_response(
                {"status": "success", "message": "Message received successfully"}
            )
        except Exception as e:
            logger.error(f"Error handling webhook request: {str(e)}")
            return web.json_response(
                {"status": "error", "message": str(e)}, status=500
            )

    async def start(self) -> None:
        """Start the mock server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Mock Google Chat server running at http://{self.host}:{self.port}")
        logger.info(f"Use this URL as your webhook URL for testing: http://{self.host}:{self.port}")

    def get_webhook_url(self) -> str:
        """Get the webhook URL for this mock server."""
        return f"http://{self.host}:{self.port}"

    def get_received_messages(self) -> list:
        """Get the list of received messages."""
        return self.received_messages


async def run_mock_server():
    """Run the mock server."""
    server = MockGoogleChatServer()
    await server.start()
    
    # Keep the server running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_mock_server())
