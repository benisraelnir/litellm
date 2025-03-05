"""
Example script demonstrating how to use Google Chat notifications in a data pipeline.

This script simulates a data processing pipeline and sends notifications about
job completion status to a Google Chat space.
"""

import asyncio
import random
import time
from typing import Any, Dict, Optional

# Import the Google Chat notification functionality
from litellm.integrations.GoogleChatAlerting import (
    get_webhook_url_from_env,
    send_chat_notification,
)


async def process_data(job_id: str) -> Dict[str, Any]:
    """
    Simulate processing data in a pipeline.

    Args:
        job_id: The ID of the job.

    Returns:
        Dict containing job results.
    """
    print(f"Starting data processing job {job_id}...")

    # Simulate processing time
    processing_time = random.uniform(1.0, 3.0)
    time.sleep(processing_time)

    # Simulate success or failure (90% success rate)
    success = random.random() < 0.9

    # Generate random job statistics
    records_processed = random.randint(1000, 10000)
    errors = 0 if success else random.randint(1, 100)

    return {
        "job_id": job_id,
        "success": success,
        "processing_time": processing_time,
        "records_processed": records_processed,
        "errors": errors,
    }


async def notify_job_completion(
    webhook_url: str, job_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send a job completion notification to Google Chat.

    Args:
        webhook_url: The webhook URL for the Google Chat space.
        job_results: The results of the job.

    Returns:
        Dict containing notification result.
    """
    job_id = job_results["job_id"]
    success = job_results["success"]
    status = "SUCCESS" if success else "FAILURE"

    # Create the message
    message = "🔔 *Data Pipeline Job Completion*\n\n"
    message += f"Job ID: {job_id}\n"
    message += f"Status: {status}\n"

    # Add job details
    message += "\n*Job Details:*\n"
    message += f"• Processing Time: {job_results['processing_time']:.2f} seconds\n"
    message += f"• Records Processed: {job_results['records_processed']}\n"
    message += f"• Errors: {job_results['errors']}\n"

    # Add a timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message += f"\n_Notification sent at {timestamp}_"

    # Send the notification
    return await send_chat_notification(
        webhook_url=webhook_url,
        message=message,
    )


async def run_pipeline(webhook_url: Optional[str] = None) -> None:
    """
    Run a simulated data pipeline with Google Chat notifications.

    Args:
        webhook_url: The webhook URL for the Google Chat space.
            If not provided, will try to get it from environment variables.
    """
    # If webhook_url is not provided, try to get it from environment variables
    if not webhook_url:
        webhook_url = get_webhook_url_from_env()
        if not webhook_url:
            print(
                "Error: No webhook URL provided and GOOGLE_CHAT_WEBHOOK_URL environment variable not set."
            )
            print(
                "Please provide a webhook URL or set the GOOGLE_CHAT_WEBHOOK_URL environment variable."
            )
            return

    # Generate a random job ID
    import uuid

    job_id = f"job-{uuid.uuid4().hex[:8]}"

    # Process data
    job_results = await process_data(job_id)

    # Send notification
    notification_result = await notify_job_completion(webhook_url, job_results)

    # Print results
    if notification_result.get("success", False):
        print(f"✅ Job completion notification sent successfully for job {job_id}")
    else:
        print(
            f"❌ Failed to send job completion notification: {notification_result.get('message', 'Unknown error')}"
        )


if __name__ == "__main__":
    # Get webhook URL from command line argument or environment variable
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a simulated data pipeline with Google Chat notifications"
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        help="The webhook URL for the Google Chat space. If not provided, will try to get it from GOOGLE_CHAT_WEBHOOK_URL environment variable.",
    )
    args = parser.parse_args()

    # Run the pipeline
    asyncio.run(run_pipeline(webhook_url=args.webhook_url))
