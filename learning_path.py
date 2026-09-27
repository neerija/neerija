import os
import traceback
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.Client(model_name="models/gemini-2.5-flash")


def get_learning_recommendations(topic: str) -> str:
    prompt = f"""You are an AI tutor. The student wants to learn about: {topic}.
Suggest a structured and adaptive learning path including key topics, order of learning, and resources.
Include beginner, intermediate, and advanced levels if needed.
"""
    try:
        response = model.generate_content(prompt)
        print("🧠 Gemini raw response:", response)

        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "parts") and response.parts:
            return response.parts[0].text
        else:
            return "❌ Could not extract content from Gemini response."
    except Exception as e:
        traceback.print_exc()  # This gives full error trace
        return f"❌ Error occurred: {str(e)}"