# 🤖 iSMRITI — NLP Query-Based Conversational Chatbot

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Deep%20Learning-orange.svg)](https://github.com/tripathiswastik/firstchatbot)
[![Author](https://img.shields.io/badge/Author-Swastik%20Tripathi-blueviolet.svg)](https://github.com/tripathiswastik)

An intelligent Question-Answering conversational agent designed to map user queries to knowledge-base responses using modern Natural Language Processing (NLP) and vector similarity.

---

## 🌟 Key Features

- **⚡ Lightweight TF-IDF Engine (`chatbot.py`)**:
  - Pure Python standard library implementation (no heavy dependencies required).
  - Fast cosine similarity scoring with sub-millisecond response latency.
  - Returns answer confidence score and closest matched question.
  - Automatically logs unhandled questions to `unansweredquestions.txt` for continuous training.
- **🧠 Neural Network Model (`chatbot_query.py`)**:
  - Multi-layer Keras/TensorFlow Sequential classifier with multi-hot question representations.
  - Integrated Tkinter GUI dialog interface (`Welcome to iSMRITI Chatbot`).
- **📁 Portable File Handling**:
  - Dynamically detects dataset files (`question.txt`, `answer.txt`) using relative path resolution (fixing broken hardcoded desktop paths).

---

## 📂 Project Structure

```text
firstchatbot/
├── chatbot.py                 # Fast, dependency-free TF-IDF similarity chatbot (Recommended)
├── chatbot_query.py           # Deep Learning (Keras/TensorFlow) model with Tkinter GUI
├── question.txt               # Knowledge base input questions dataset
├── answer.txt                 # Knowledge base corresponding answers dataset
├── unansweredquestions.txt    # Audit log of unrecognized queries for retraining
├── requirements.txt           # Optional deep learning dependencies
├── LICENSE                    # GNU General Public License v3.0
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### Option 1: Run Lightweight Engine (Instant — No Installation Required)
Run out-of-the-box using standard Python:

```bash
python chatbot.py
```

**Example Conversation:**
```text
============================================================
         🤖 iSMRITI AI Chatbot (TF-IDF NLP Engine)
============================================================
Loaded 99 question-answer pairs successfully!

You: What is AI?
ChatBot: Artificial Intelligence is the branch of engineering and science devoted to constructing machines that think.
         [Confidence: 100.0% | Match: 'What is AI?']

You: Are you sentient?
ChatBot: By the strictest dictionary definition of the word 'sentience', I may be
         [Confidence: 100.0% | Match: 'Are you sentient?']
```

---

### Option 2: Run Deep Learning Model with GUI (`chatbot_query.py`)

If you want to train the Keras neural network and open the Tkinter GUI:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch GUI:**
   ```bash
   python chatbot_query.py
   ```

---

## 📊 How It Works

1. **Text Normalization**: Expands common contractions (e.g., `what's` $\rightarrow$ `what is`), converts text to lowercase, and strips extraneous punctuation.
2. **Vector Space Encoding**: Computes term frequency and inverse document frequency (TF-IDF) across the corpus.
3. **Similarity Scoring**: Calculates cosine similarity against all known question vectors.
4. **Fallback Mechanism**: Queries with confidence below threshold are logged to `unansweredquestions.txt` to help developers expand the bot's knowledge base.

---

## 👥 Contributors

- **Swastik Tripathi** ([@tripathiswastik](https://github.com/tripathiswastik))
- Aviral, Bharat, Priyanshu

---

## 📄 License

This repository is licensed under the [GNU General Public License v3.0](LICENSE).
