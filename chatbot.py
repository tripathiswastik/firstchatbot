#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart TF-IDF & Cosine Similarity Lightweight Chatbot Engine
Requires zero heavyweight dependencies (pure Python standard library).
"""

import math
import os
import re
from collections import Counter


def clean_text(text: str) -> str:
    """Preprocess, expand contractions, and remove punctuation from query."""
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text.strip()


class LightChatbot:
    def __init__(self, questions_path="question.txt", answers_path="answer.txt"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        q_file = os.path.join(base_dir, questions_path)
        a_file = os.path.join(base_dir, answers_path)

        with open(q_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_questions = [line.strip() for line in f if line.strip()]

        with open(a_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_answers = [line.strip() for line in f if line.strip()]

        # Pair questions and answers up to the available minimum count
        total_pairs = min(len(raw_questions), len(raw_answers))
        self.corpus = []
        for i in range(total_pairs):
            cleaned_q = clean_text(raw_questions[i])
            self.corpus.append({
                "raw_q": raw_questions[i],
                "cleaned_q": cleaned_q,
                "tokens": cleaned_q.split(),
                "answer": raw_answers[i]
            })

        self.num_docs = len(self.corpus)
        self.idf = self._compute_idf()
        self.doc_vectors = [self._compute_tfidf_vector(doc["tokens"]) for doc in self.corpus]

    def _compute_idf(self):
        doc_freq = Counter()
        for doc in self.corpus:
            unique_tokens = set(doc["tokens"])
            for token in unique_tokens:
                doc_freq[token] += 1

        idf = {}
        for token, count in doc_freq.items():
            idf[token] = math.log((self.num_docs + 1) / (count + 1)) + 1.0
        return idf

    def _compute_tfidf_vector(self, tokens):
        tf = Counter(tokens)
        vector = {}
        for token, count in tf.items():
            idf_val = self.idf.get(token, math.log(self.num_docs + 1) + 1.0)
            vector[token] = count * idf_val
        return vector

    def _cosine_similarity(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[x] * vec2[x] for x in intersection)

        norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

        if not norm1 or not norm2:
            return 0.0
        return dot_product / (norm1 * norm2)

    def respond(self, query: str):
        cleaned_query = clean_text(query)
        query_tokens = cleaned_query.split()

        if not query_tokens:
            return {
                "answer": "Please ask a question.",
                "confidence": 0.0,
                "matched_question": None
            }

        query_vector = self._compute_tfidf_vector(query_tokens)

        best_score = -1.0
        best_match = None

        for idx, doc_vec in enumerate(self.doc_vectors):
            sim = self._cosine_similarity(query_vector, doc_vec)
            if sim > best_score:
                best_score = sim
                best_match = self.corpus[idx]

        if best_score < 0.25:
            # Log to unanswered questions file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base_dir, "unansweredquestions.txt"), "a", encoding="utf-8") as f:
                f.write(query + "\n")

            return {
                "answer": "I'm not quite sure how to answer that yet. Could you rephrase?",
                "confidence": round(best_score, 4),
                "matched_question": None
            }

        return {
            "answer": best_match["answer"],
            "confidence": round(best_score, 4),
            "matched_question": best_match["raw_q"]
        }


def main():
    print("=" * 60)
    print("         🤖 iSMRITI AI Chatbot (TF-IDF NLP Engine)")
    print("=" * 60)
    print("Type your question below. Type 'bye' or 'exit' to quit.\n")

    bot = LightChatbot()
    print(f"Loaded {bot.num_docs} question-answer pairs successfully!\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("bye", "exit", "quit", "shutup", "get lost"):
                print("ChatBot: Goodbye! Have a wonderful day!")
                break

            response = bot.respond(user_input)
            print(f"ChatBot: {response['answer']}")
            print(f"         [Confidence: {response['confidence'] * 100:.1f}% | Match: '{response['matched_question']}']\n")

        except (KeyboardInterrupt, EOFError):
            print("\nChatBot: Session closed.")
            break


if __name__ == "__main__":
    main()
