# Google Chat Alerting for LiteLLM

This module provides functionality to send notifications to Google Chat spaces using webhooks.

## Usage

```python
import asyncio
from litellm.integrations.GoogleChatAlerting import send_chat_notification

async def send_notification():
    webhook_url = "https://chat.googleapis.com/v1/spaces/SPACE_ID/messages?key=KEY&token=TOKEN"
    message = "Hello from LiteLLM!"
    
    result = await send_chat_notification(
        webhook_url=webhook_url,
        message=message
    )
    
    print(f"Notification sent: {result}")

# Run the async function
asyncio.run(send_notification())
```

## Environment Variables

You can set the Google Chat webhook URL as an environment variable:

```
GOOGLE_CHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/SPACE_ID/messages?key=KEY&token=TOKEN
```

Then retrieve it using:

```python
from litellm.integrations.GoogleChatAlerting import get_webhook_url_from_env

webhook_url = get_webhook_url_from_env()
```

## Features

- Send simple text messages to Google Chat spaces
- Support for threading messages
- Error handling for failed message delivery
- Async implementation for non-blocking operation
