from typing import Any, Literal, Optional, cast
import os
import requests
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
import time
from dsp.modules.lm import LM

try:
    import mlflow.deployments
except ImportError:
    pass

class Databricks_MLFlow(LM):
    def __init__(self, client=None, model=None, **kwargs):
        self.client = client
        self.provider = "databricks"
        self.history: list[dict[str, Any]] = []
        self.model = model
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 2048,
            "stop": [],
            "use_raw_prompt": True,
            **kwargs,
        }

    def basic_request(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        kwargs["messages"] = [{
              "role": "user",
              "content": "Hello!"
            },
            {
              "role": "assistant",
              "content": "Hello! How can I assist you today?"
            }, 
            {"role": "user", "content": prompt}]

        response = self.client.predict(
            endpoint=self.model,
            inputs=kwargs
        )
        if not response or 'choices' not in response:
            raise Exception("Error hitting endpoint")
        else:
            completions = response["choices"]
            response_text = (
                completions[0].get("message").get("content")
            )
            history = {
                "prompt": prompt,
                "response": response_text,
                "kwargs": kwargs,
            }
            self.history.append(history)

            return [response_text]

    def request(self, prompt: str, **kwargs):
        return self.basic_request(prompt, **kwargs)
        
    def __call__(self, prompt: str, **kwargs):
        return self.request(prompt, **kwargs)