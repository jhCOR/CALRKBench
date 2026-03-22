from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple
import random
import numpy as np
from openai import OpenAI
from google import genai
from eval_utils import load_json
from google import genai
from google.genai import types

### API Setting
DEFAULT_SECRET_PATH = "/home/elicer/SECRETE/key.json"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

### Model List
MODEL_GPT = "gpt-5-2025-08-07"
MODEL_GEMINI = "gemini-3-flash-preview" 
MODEL_SOLAR = "solar-pro3-260126"

### Setting
FIXED_TEMPERATURE = 0
FIXED_TOP_P = 1
FIXED_SEED = 42
random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)

def _extract_chat_text(resp: Any) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for part in content:
            text = None
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None)
                if text is None:
                    text = getattr(part, "content", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n".join(chunks).strip()

    if content is None:
        return ""
    return str(content).strip()

class GptClient:
    # https://developers.openai.com/api/docs/guides/reasoning
    def __init__(self, client: OpenAI,
        reasoning_effort: Optional[str],
        reasoning_enabled: bool,
        model_id:str
    ):  
        self.client = client
        if reasoning_enabled is False:
            reasoning_effort = "none"
        self.setting = {
            "reasoning_effort":reasoning_effort,
            "model": model_id,
            # "temperature": FIXED_TEMPERATURE,
            "top_p": FIXED_TOP_P,
            "seed": FIXED_SEED
        }
        print("GptClient", self.setting, flush=True)

    def call(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        text = ""
        try:
            
            self.setting["messages"] = [{"role": "user", "content": prompt}]
            resp = self.client.chat.completions.create(**self.setting)
            text = _extract_chat_text(resp)

            return text, {
                "ok": True,
                "used_reasoning": self.setting["reasoning_effort"],
                "model_runner": self.setting["model"],
            }
        except Exception as exc:
            print("Error: ", exc)
            return text, {
                "ok": False,
                "error":exc, 
                "used_reasoning": self.setting["reasoning_effort"],
                "model_runner": self.setting["model"],
            }

class SolarPro3Client:
    # Solar Pro 3 : low means no reasoning,  medium / high: reasoning ON

    def __init__(self, client: OpenAI,
        reasoning_effort: Optional[str],
        reasoning_enabled: bool,
        model_id:str
    ):
        self.client = client
        reasoning_effort = reasoning_effort if reasoning_enabled is True else "low"
        self.setting = {
            "reasoning_effort":reasoning_effort, 
            "model": model_id,
            "temperature": FIXED_TEMPERATURE,
            "top_p": FIXED_TOP_P,
            "seed": FIXED_SEED,
        }
        print("SolarPro3Client", self.setting, flush=True)
    
    def call(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        text = ""
        try:
            self.setting["messages"] = [{"role": "user", "content": prompt}]
            resp = self.client.chat.completions.create(**self.setting)
            text = _extract_chat_text(resp)

            return text, {
                "ok": True,
                "used_reasoning": self.setting["reasoning"],
                "model_runner": self.setting["model"],
            }
        except Exception as exc:
            print("Error: ", exc)
            return text, {
                "ok": False,
                "error":exc, 
                "used_reasoning": self.setting["reasoning"],
                "model_runner": self.setting["model"],
            }

class GeminiClient:
    # https://ai.google.dev/gemini-api/docs/text-generation?hl=ko
    # Gemini 3.1 Pro thinkingLevel : low / medium / high
    # Gemini 3 Flash thinkingLevel : minimal / low / medium / high
    def __init__(self, client: OpenAI,
        reasoning_effort: Optional[str],
        reasoning_enabled: bool,
        model_id:str
    ):
        self.client = client
        think_setting = types.ThinkingConfig(thinking_level=reasoning_effort) if reasoning_enabled is True else types.ThinkingConfig(thinking_budget=0)
        config=types.GenerateContentConfig(
            thinking_config=think_setting,
            temperature=FIXED_TEMPERATURE,
            topP = FIXED_TOP_P,
            seed = FIXED_SEED
        )
        self.setting = {
            "model": model_id,
            "config": config
        }
        print("GeminiClient", self.setting, flush=True)

    def call(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        text = ""
        try:
            self.setting["contents"] = prompt
            resp = self.client.models.generate_content(**self.setting)

            return resp.text, {
                "ok": True,
                "used_reasoning": str(self.setting["config"].thinking_config.thinking_level),
                "model_runner": self.setting["model"],
            }
        except Exception as exc:
            print("Error: ", exc)
            return text, {
                "ok": False,
                "error":exc, 
                "used_reasoning": str(self.setting["config"].thinking_config.thinking_level),
                "model_runner": self.setting["model"],
            }

class GenericModelClient:
    def __init__(self, client: OpenAI,
        reasoning_effort: Optional[str],
        reasoning_enabled: bool,
        model_id:str
    ):
        self.client = client
        self.model_id = model_id
        think_setting = {"effort": reasoning_effort} if reasoning_enabled is True else None
        self.setting = {
            "model": model_id,
            "temperature": FIXED_TEMPERATURE,
            "top_p": FIXED_TOP_P,
            "seed": FIXED_SEED,
            "extra_body":{"reasoning": think_setting}
        }
        print("\nGenericModelClient", self.setting, flush=True)
    def call(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        text = ""
        try:
            self.setting["messages"] = [{"role": "user", "content": prompt}]
            resp = self.client.chat.completions.create(**self.setting)
            text = _extract_chat_text(resp)
            return text, {
                "ok": True,
                "used_reasoning": self.setting["extra_body"]["reasoning"],
                "model_runner": "generic",
            }
        except Exception as exc:
            print("Error: ", exc)
            return text, {
                "ok": True,
                "error":exc, 
                "used_reasoning": self.setting["extra_body"]["reasoning"],
                "model_runner": "generic",
            }

def call_llm(
    model: str,
    service: str,
    prompt: str,
    key:str,
    reasoning_effort: Optional[str] = None,
    reasoning_enabled: bool = False,
    
    max_retries: int = 1,
    sleep_base: float = 1.5
) -> Tuple[str, Dict[str, Any]]:
    secret = load_json(DEFAULT_SECRET_PATH)
    api_key = secret.get(key)

    if service == "openrouter":
        headers: Dict[str, str] = {}

        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": DEFAULT_OPENROUTER_BASE_URL,
        }

        if headers:
            kwargs["default_headers"] = headers
        client = OpenAI(**kwargs)
        llm_runner = GenericModelClient(client, reasoning_effort, reasoning_enabled, model)
    else:
        if model == MODEL_SOLAR:
            client = OpenAI(
                api_key=api_key,
                base_url=DEFAULT_UPSTAGE_BASE_URL,
            )
            llm_runner = SolarPro3Client(client, reasoning_effort, reasoning_enabled, model)

        if model == MODEL_GPT:
            client = OpenAI(
                api_key=api_key
            )
            llm_runner = GptClient(client, reasoning_effort, reasoning_enabled, model)

        if model == MODEL_GEMINI:
            client = genai.Client(api_key=api_key)
            llm_runner = GeminiClient(client, reasoning_effort, reasoning_enabled, model)

    response = llm_runner.call(prompt)
    print(response, flush=True)
    return response