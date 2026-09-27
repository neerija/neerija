import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and silence tied weights warning
explain_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
explain_model = AutoModelForSeq2SeqLM.from_pretrained(
    "MBZUAI/LaMini-Flan-T5-783M",
    tie_word_embeddings=False,
)


def explain_topic(topic: str) -> str:
    input_text = f"Explain the concept of '{topic}' in a simple and clear way for a school student."
    inputs = explain_tokenizer(input_text, return_tensors="pt")

    # Generate explanation without tracking gradients to save memory
    with torch.no_grad():
        outputs = explain_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

    explanation = explain_tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )
    return explanation