import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()


def summarize_text(text: str) -> str:
    try:
        model = genai.Client(model_name="models/gemini-2.5-flash")
        prompt = f"Summarize the following text in simple language:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error in Summary: {e}"