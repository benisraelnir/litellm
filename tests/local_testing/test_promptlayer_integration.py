import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import time

import pytest

import litellm
from litellm import completion

# def test_promptlayer_logging():
#     try:
#         # Redirect stdout
#         old_stdout = sys.stdout
#         sys.stdout = new_stdout = io.StringIO()


#         response = completion(model="claude-3-5-haiku-20241022",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm claude"
#                               }])

#         # Restore stdout
#         time.sleep(1)
#         sys.stdout = old_stdout
#         output = new_stdout.getvalue().strip()
#         print(output)
#         if "LiteLLM: Prompt Layer Logging: success" not in output:
#             raise Exception("Required log message not found!")

#     except Exception as e:
#         print(e)

# test_promptlayer_logging()


@pytest.mark.skip(
    reason="this works locally but fails on ci/cd since ci/cd is not reading the stdout correctly"
)
def test_promptlayer_logging_with_metadata():
    try:
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()
        litellm.set_verbose = True
        litellm.success_callback = ["promptlayer"]

        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm ai21"}],
            temperature=0.2,
            max_tokens=20,
            metadata={"model": "ai21"},
        )

        # Restore stdout
        time.sleep(1)
        sys.stdout = old_stdout
        output = new_stdout.getvalue().strip()
        print(output)

        assert "Prompt Layer Logging: success" in output

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(
    reason="this works locally but fails on ci/cd since ci/cd is not reading the stdout correctly"
)
def test_promptlayer_logging_with_metadata_tags():
    try:
        # Redirect stdout
        litellm.set_verbose = True

        litellm.success_callback = ["promptlayer"]
        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()

        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm ai21"}],
            temperature=0.2,
            max_tokens=20,
            metadata={"model": "ai21", "pl_tags": ["env:dev"]},
            mock_response="this is a mock response",
        )

        # Restore stdout
        time.sleep(1)
        sys.stdout = old_stdout
        output = new_stdout.getvalue().strip()
        print(output)

        assert "Prompt Layer Logging: success" in output
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# def test_chat_openai():
#     try:
#         response = completion(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm openai"
#                               }])

#         print(response)
#     except Exception as e:
#         print(e)

# test_chat_openai()
