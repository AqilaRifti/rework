import json
import requests
from typing import List, Dict, Optional

FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MISTRAL_8X22B_INSTRUCT = "accounts/fireworks/models/mixtral-8x22b-instruct"

def get_fireworks_default_headers(api_key: str):
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

def get_fireworks_payload(
        model: str, 
        max_tokens: int, 
        top_p: int, 
        top_k: int, 
        presence_penalty: float,
        frequency_penalty: float,
        temperature: float,
        messages: List[Dict[str, str]]
    ):
    return {
        "model": model,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "temperature": temperature,
        "messages": messages
    }

def fireworks_chat_provider(
        model: str, 
        max_tokens: int, 
        top_p: int, 
        top_k: int, 
        presence_penalty: float,
        frequency_penalty: float,
        temperature: float,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = "json"
    ):
    resp = requests.request(
        "POST", 
        FIREWORKS_CHAT_URL, 
        headers=get_fireworks_default_headers, 
        data=json.dumps(get_fireworks_payload(
            model,
            max_tokens,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            temperature,
            messages 
        ))
    )
    if response_format == "json":
        return resp.json()
    elif response_format == "raw":
        return resp.raw
    elif response_format == "text":
        return resp.text
    else:
        raise Exception("Format Error: Unavailable Format")
