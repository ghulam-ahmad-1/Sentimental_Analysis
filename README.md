# 🧠 RoBERTa-Based Sentiment Analysis

A simple command-line tool for performing sentiment analysis using the pretrained **`cardiffnlp/twitter-roberta-base-sentiment`** model from Hugging Face. Optimized for analyzing social media text, short reviews, and comments.

---

## 📌 Features

- Utilizes state-of-the-art pretrained RoBERTa transformer
- Handles noisy text with user mentions and URLs
- Predicts sentiment as **Positive**, **Neutral**, or **Negative**
- Outputs probability scores for interpretability

---

## 🔍 How It Works

1. **Input**: User provides a comment or review via terminal
2. **Preprocessing**: Cleans mentions and links
3. **Encoding**: Uses RoBERTa tokenizer
4. **Prediction**: Model outputs class scores processed via `softmax`
5. **Output**: Displays score per sentiment and predicted class

---

## 🧪 Example

```bash
Enter the Comment Or Review  : I love this product! Works perfectly.
Negative 0.02
Neutral 0.15
Positive 0.83
The Sentiment of the Comment is : Positive
