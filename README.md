## 🧠 Question Answering System
This project is a Natural Language Processing (NLP) application that answers questions based on a provided passage or context using fine-tuned transformer models (e.g., BERT, RoBERTa, DistilBERT).

---

## 🧩 Key Steps
- **Dataset:** Used the SQuAD v1.1 (Stanford Question Answering Dataset) for training and evaluation.
- **Model Selection:** Tested multiple transformer models fine-tuned for QA:
  - DistilBERT (distilbert-base-uncased-distilled-squad)
  - BERT-Large (bert-large-uncased-whole-word-masking-finetuned-squad)
  - RoBERTa (deepset/roberta-base-squad2)
  - ALBERT (twmkn9/albert-base-v2-squad2)
- **Answer Extraction:** The model receives both the context and question, predicts start and end token positions, and extracts the answer span. 
- **Evaluation Metrics:** Measured performance using Exact Match (EM) and F1 Score.
- **Deployment:** Built a simple Streamlit app for interactive question answering.
---

## 📂 Dataset
The dataset used is the SQuAD v1.1 – Stanford Question Answering Dataset (100,000+ samples).
- Available on:
  - [Kaggle - Stanford Question Answering Dataset](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset)
  - [Google Drive - Stanford Question Answering Dataset](https://drive.google.com/drive/folders/1fE0o6XC-nPR71Ne_fxNYq2yoeuSCTJys?usp=drive_link)

---

## 📊 Model Performance Comparison
| Model          | Exact Match (EM) |  F1 Score | Remarks             |
| :------------- | :--------------: | :-------: | :------------------ |
| **DistilBERT** |       75.00      |   82.26   | Lightweight & Fast  |
| **BERT-Large** |       83.00      |   88.97   | Strong baseline     |
| **RoBERTa**    |     **94.00**    | **95.23** | 🏆 Best performance |
| **ALBERT**     |       72.00      |   77.99   | Most compact        |

---

## 🧠 Tech Stack & Tools
- **Python Libraries**:  
  `Pandas`, `Evaluate`, `PyTorch`, `Transformers` 
- **Deployment**: Streamlit for interactive prediction  
- **Others**: GitHub / Google Colab / Kaggle for experimentation

---

## 📦 Dependencies
Before running locally, ensure these are installed:

```sh
pip install pandas evaluate torch transformers streamlit joblib
```

## Installing
To install Streamlit:
```sh
pip install streamlit
```
To install all required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Application Locally
```sh
streamlit run app.py
```
Then open the local URL (usually http://localhost:8501/) in your browser.

## 📰 Try the App Online
You can use the app directly here: [ask-me-ai](https://ask-me-ai.streamlit.app/)<br>
Simply paste a context passage, type a question, and click Get Answer to see the model’s response.

---

## 💡 Features
- Accepts long passages or paragraphs as input
- Answers questions directly from the provided context
- Uses multiple fine-tuned transformer models
- Displays accurate, human-like responses
- Fully deployed via Streamlit

---

## 📂 Folder Structure
```
Question-Answering-with-Transformers/
├── app.py                    
├── qa_model/                 
├── qa_tokenizer/             
├── requirements.txt           
└── README.md

```

## ❓ Help
If you encounter any issues:
- Check the [Streamlit Documentation](https://docs.streamlit.io/)
- Search for similar issues or solutions on [Kaggle](https://www.kaggle.com/)
- Open an issue in this repository

---

## ✍️ Author
👤 Oluyale Ezekiel
- 📧 Email: ezekieloluyale@gmail.com
- LinkedIn: [Ezekiel Oluyale](https://www.linkedin.com/in/ezekiel-oluyale)
- GitHub Profile: [@amusEcode1](https://github.com/amusEcode1)
- Twitter: [@amusEcode1](https://x.com/amusEcode1?t=uHxhLzrA1TShRiSMrYZQiQ&s=09)

---

## 🙏 Acknowledgement
Thank you, Elevvo, for the incredible opportunity and amazing Internship.
