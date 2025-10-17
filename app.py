!pip install streamlit -q
!pip install transformers -q
!pip install torch -q

import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from google.colab import drive
drive.mount('/content/drive')

# Load model + tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained("/content/drive/MyDrive/Models/Question_Answering/qa_model")
    tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/Models/Question_Answering/qa_tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

def get_answer(context, question):
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
    return answer

st.set_page_config(page_title="Question-Answering", page_icon="🧠", layout="wide")
st.title("🧠 Question Answering App")
st.markdown("Ask questions based on a passage using a fine-tuned QA model.")

context = st.text_area("📄 Enter a passage:", height=200)
question = st.text_input("❓ Enter your question:")

if st.button("Get Answer"):
    if context and question:
        answer = get_answer(context, question)
        st.success(f"**Answer:** {answer}")
    else:
        st.warning("Please enter both context and question.")

!jupyter nbconvert --to script "/content/drive/MyDrive/Colab Notebooks/Question_Answering_app/app.ipynb"

!mv "/content/drive/MyDrive/Colab Notebooks/Question_Answering_app/app.txt" "/content/drive/MyDrive/Colab Notebooks/Question_Answering_app/app.py"
