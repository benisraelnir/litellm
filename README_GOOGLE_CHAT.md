# Google Chat Notifications for LiteLLM

This document provides step-by-step instructions for setting up Google Chat notifications for your data pipeline using LiteLLM. These notifications can be used to send job completion alerts and other important messages from your automated processes.

## Table of Contents

1. [Overview](#overview)
2. [Creating a Google Chat Webhook](#creating-a-google-chat-webhook)
3. [Required GCP API Enablement](#required-gcp-api-enablement)
4. [Service Account Setup](#service-account-setup)
5. [Storing Credentials Securely](#storing-credentials-securely)
6. [Using the Google Chat Notification Function](#using-the-google-chat-notification-function)
7. [Troubleshooting](#troubleshooting)

## Overview

Google Chat notifications in LiteLLM allow you to send messages to Google Chat spaces using webhooks. This is particularly useful for:

- Sending job completion notifications from data pipelines
- Alerting teams about errors or issues
- Providing status updates for long-running processes
- Notifying about budget thresholds or other important events

The implementation uses webhook URLs, which provide a simple, non-interactive way to authenticate and send messages to Google Chat spaces.

## Creating a Google Chat Webhook

To create a webhook for Google Chat:

1. **Open Google Chat**:
   - Go to [Google Chat](https://chat.google.com/)
   - Sign in with your Google Workspace account

2. **Create or Select a Space**:
   - Create a new space or select an existing one where you want to receive notifications
   - You must have appropriate permissions in the space to create webhooks

3. **Access Space Settings**:
   - Click on the space name at the top
   - Select "Manage webhooks" from the dropdown menu

4. **Add a Webhook**:
   - Click "Add webhook"
   - Enter a name for your webhook (e.g., "LiteLLM Pipeline Notifications")
   - Optionally, add an avatar URL for the webhook
   - Click "Save"

5. **Copy the Webhook URL**:
   - After creating the webhook, copy the provided URL
   - This URL contains the necessary authentication information
   - Keep this URL secure as anyone with this URL can post messages to your space

## Required GCP API Enablement

For Google Chat webhooks, you typically don't need to enable additional GCP APIs since webhooks use a simple HTTPS endpoint. However, if you're using other Google Cloud services alongside your notifications, you may need to enable specific APIs.

If you're building a more complex integration that requires the Chat API:

1. **Go to the Google Cloud Console**:
   - Visit [Google Cloud Console](https://console.cloud.google.com/)

2. **Select Your Project**:
   - Choose the project you want to use for your integration

3. **Navigate to APIs & Services**:
   - Go to "APIs & Services" > "Library"

4. **Enable the Google Chat API**:
   - Search for "Google Chat API"
   - Click on it and then click "Enable"

## Service Account Setup

For simple webhook-based notifications, a service account is not required. The webhook URL itself contains the necessary authentication information.

However, if you're building a more complex integration that requires the Chat API, follow these steps to set up a service account:

1. **Create a Service Account**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Enter a name and description for the service account
   - Click "Create and Continue"

2. **Assign Roles**:
   - For Chat API access, assign the "Chat API User" role
   - Click "Continue"

3. **Grant Users Access**:
   - Optionally, grant users access to the service account
   - Click "Done"

4. **Create a Key**:
   - Find your service account in the list
   - Click on the three dots menu and select "Manage keys"
   - Click "Add Key" > "Create new key"
   - Choose JSON as the key type
   - Click "Create"
   - The key file will be downloaded to your computer

## Storing Credentials Securely

For webhook URLs and service account credentials, it's important to store them securely:

### Environment Variables

Store the webhook URL as an environment variable:

```bash
# For webhook URL
export GOOGLE_CHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/SPACE_ID/messages?key=KEY&token=TOKEN"

# For service account (if using the Chat API)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Secret Management Services

For production environments, use a secret management service:

1. **Google Secret Manager**:
   ```bash
   # Store the secret
   echo -n "https://chat.googleapis.com/v1/spaces/SPACE_ID/messages?key=KEY&token=TOKEN" | \
   gcloud secrets create google-chat-webhook --data-file=-
   
   # Access the secret in your application
   WEBHOOK_URL=$(gcloud secrets versions access latest --secret=google-chat-webhook)
   ```

2. **Environment-specific Configuration Files**:
   - Create separate configuration files for different environments
   - Ensure these files are not committed to version control
   - Use a `.env` file with a package like `python-dotenv`

   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   webhook_url = os.getenv("GOOGLE_CHAT_WEBHOOK_URL")
   ```

3. **Kubernetes Secrets**:
   If you're running in Kubernetes:
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: google-chat-credentials
   type: Opaque
   data:
     webhook_url: BASE64_ENCODED_WEBHOOK_URL
   ```

   Then mount as environment variables in your pod.

## Using the Google Chat Notification Function

Here's how to use the Google Chat notification function in your code:

```python
import asyncio
from litellm.integrations.GoogleChatAlerting import send_chat_notification

async def notify_job_completion(job_id, status, details=None):
    """
    Send a job completion notification to Google Chat.
    
    Args:
        job_id: The ID of the completed job
        status: The status of the job (success, failure, etc.)
        details: Optional details about the job
    """
    # Get the webhook URL from environment variables
    from litellm.integrations.GoogleChatAlerting import get_webhook_url_from_env
    webhook_url = get_webhook_url_from_env()
    
    if not webhook_url:
        print("Google Chat webhook URL not configured")
        return
    
    # Create the message
    message = f"🔔 *Job Completion Notification*\n\n"
    message += f"Job ID: {job_id}\n"
    message += f"Status: {status}\n"
    
    if details:
        message += "\n*Details:*\n"
        for key, value in details.items():
            message += f"• {key}: {value}\n"
    
    # Send the notification
    result = await send_chat_notification(
        webhook_url=webhook_url,
        message=message
    )
    
    return result

# Example usage
async def main():
    job_details = {
        "duration": "45 minutes",
        "records_processed": 10000,
        "errors": 0
    }
    
    result = await notify_job_completion(
        job_id="data-pipeline-123",
        status="SUCCESS",
        details=job_details
    )
    
    print(f"Notification sent: {result}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues and Solutions

1. **Webhook URL Not Working**:
   - Verify the webhook URL is correct and hasn't expired
   - Ensure the space still exists and the webhook hasn't been removed
   - Check if you have the necessary permissions in the space

2. **Messages Not Appearing**:
   - Verify your network can reach Google Chat (no firewall issues)
   - Check for rate limiting (too many messages in a short time)
   - Ensure your message payload is correctly formatted

3. **Authentication Errors**:
   - For webhook URLs, ensure the URL is complete and hasn't been truncated
   - For service accounts, verify the key file is accessible and valid
   - Check that the service account has the necessary permissions

4. **HTTP Errors**:
   - 400 errors: Check your message format
   - 403 errors: Authentication or permission issues
   - 429 errors: You're being rate limited

### Debugging Tips

- Enable verbose logging in LiteLLM to see detailed request and response information
- Use a tool like Postman to test your webhook URL directly
- Check the Google Chat API documentation for any recent changes or updates

For additional help, refer to the [Google Chat API documentation](https://developers.google.com/chat) or open an issue on the LiteLLM GitHub repository.
