"""
Support for o1 model family 

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- modalities: image => drop param (if user opts in to dropping param)  
- role: system ==> translate to role 'user' 
- streaming => faked by LiteLLM 
- Tools, response_format =>  drop param (if user opts in to dropping param) 
- Logprobs => drop param (if user opts in to dropping param)
- Temperature => drop param (if user opts in to dropping param)
"""

from ...openai.chat.o1_transformation import OpenAIO1Config


class AzureOpenAIO1Config(OpenAIO1Config):
    def is_o1_model(self, model: str) -> bool:
        """
        Check if the model is an O1 model, with detailed logging
        """
        o1_models = ["o1-mini", "o1-preview", "azure/chatgpt-v2", "chatgpt-v2"]
        print(f"\nO1Config: Starting model check")
        print(f"Input model: {model}")
        print(f"Known O1 models: {o1_models}")

        if model is None:
            print("O1Config: Model is None, returning False")
            return False

        # Strip any azure/ prefix for matching
        model_name = (
            model.replace("azure/", "") if model.startswith("azure/") else model
        )
        print(f"Normalized model name for comparison: {model_name}")

        for m in o1_models:
            base_model = m.replace("azure/", "") if m.startswith("azure/") else m
            print(f"Comparing with O1 model: {m} (normalized: {base_model})")

            if base_model == model_name:
                print(f"O1Config: ✓ Match found! {model} is an O1 model (matched {m})")
                return True

        print(f"O1Config: ✗ No match found. {model} is not an O1 model")
        return False
