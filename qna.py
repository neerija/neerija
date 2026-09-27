import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()


def answer_question_with_gemini(question: str) -> str:
    try:
        model = genai.Client(model_name="models/gemini-2.5-flash")
        response = client.models.generate_content(model="gemini-2.5-flash", contents=question)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error in QnA: {e}"