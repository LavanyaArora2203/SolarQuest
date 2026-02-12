# chatbot.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = "You are a helpful AI assistant."

def get_chat_response(messages):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content
