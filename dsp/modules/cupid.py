from typing import Any, Literal, Optional, cast
import os
import requests
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
import time
from dsp.modules.lm import LM

class Cupid(LM):
    def __init__(self, url=None, api_key: Optional[str] = None, tokenizer = None, retries_left: int = 0, **kwargs):
        self.url = url
        self.headers = {"Authorization": api_key, "Content-Type": "application/json"}
        self.tokenizer = tokenizer
        self.retries_left = retries_left
        if not api_key and not os.environ.get("MCLI_API_KEY"):
            raise ValueError("You must supply api_key or set environment variable MCLI_API_KEY")
        if not tokenizer:
            raise ValueError("You must supply a tokenizer")


        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 2048,
            "stop": [],
            "use_raw_prompt": True,
            **kwargs,
        }

    def _format_prompt(self, prompt):
        w_temp = self.tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant. The date is Feb 23, 2024. Your knowledge cuts off as of 12/31/2023."}, {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, stop="<|im_end|>")
        return w_temp

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        formatted_prompt = self._format_prompt(prompt)
        data = {
            "prompt": formatted_prompt,
            **kwargs,
        }
        response = send_cupid_request_v00(
            f"{self.url}/v2/completions",
            json=data,
            headers=self.headers,
            timeout = 180
        )

        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            if kwargs["retries_left"] > 0:
                print("Retrying...")
                time.sleep(5 * (6 - kwargs["retries_left"]))  # Exponential backoff
                return self._generate(
                    formatted_prompt,
                    temperature=kwargs["temperature"],
                    max_tokens=kwargs["max_tokens"],
                    retries_left=kwargs["retries_left"] - 1,
                    stop=kwargs["stop"],
                    use_raw_prompt=kwargs["use_raw_prompt"],
                )
            else:
                raise Exception("Too many retries")
        else:
            json_response = response.json()
            completions = json_response["choices"]
            response_text = completions[0]["text"].strip()  # Assuming you want the first completion's text
            response_usage = json_response["usage"]  # Assuming 'usage' is directly under the response
            return response_text
        
    def __call__(self, prompt: str, **kwargs):
        return self._generate(prompt, **kwargs)


@CacheMemory.cache
def send_cupid_request_v00(arg, **kwargs):
    return requests.post(arg, **kwargs)
