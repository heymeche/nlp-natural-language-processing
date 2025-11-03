# 🧠 Natural Language Processing (NLP) Project

This repository contains a collection of notebooks and case studies developed as part of my Master's studies. It covers foundational and advanced topics in Natural Language Processing, from corpus statistics to neural models and real-world applications.

---

## 📁 Structure

### 📚 Corpus Analysis
- [`corpus_stats_nltk.ipynb`](corpus/corpus_stats_nltk.ipynb) — Basic corpus creation and NLTK functionalities


### 🧱 Feature Extraction
- [`bag_of_words.ipynb`](feature-extraction/bag_of_words.ipynb) — Classic bag-of-words models (TF-IDF, binary, frequency)
- [`emolex_emotions.ipynb`](feature-extraction/emolex_emotions.ipynb) — Emotion detection using the NRC Emolex lexicon
- [`svm_classifier.ipynb`](feature-extraction/svm_classifier.ipynb) — Support Vector Machine classification using BoW and TF-IDF features
- [`chi_squared_tcor.ipynb`](feature-extraction/chi_squared_tcor.ipynb) — Feature selection using chi-squared test and TCor
- [`information_gain.ipynb`](word-representations/information_gain.ipynb) — Information Gain for feature ranking

### 🌌 Word Representations & Similarity
- [`word_2_vec.ipynb`](word-representations/word_2_vec.ipynb) — Word2Vec for word representation
- [`random_indexing.ipynb`](word-representations/random_indexing.ipynb) — Random indexing for word representations
- [`word_similarity_graphs.ipynb`](word-representations/word_similarity_graphs.ipynb) — Word constellations and similarity graphs

### 🧮 Language Modeling
- [`ngram_models.ipynb`](language-models/ngram_models.ipynb) — N-gram models (fixed λ, interpolated)
- [`em_and_hangman.ipynb`](language-models/em_and_hangman.ipynb) — Expectation Maximization + Hangman-style inference

### 🤖 Neural NLP
- [`word_level_models.ipynb`](neural-nlp/word_level_models.ipynb) — Word-level language models (with and without pretrained embeddings)
- [`char_level_models.ipynb`](neural-nlp/char_level_models.ipynb) — Character-level modeling
- [`gru_attention.ipynb`](neural-nlp/gru_attention.ipynb) — GRU + Hierarchical Attention networks

### 🧪 Case Studies
- [`tripadvisor_analysis.ipynb`](case-studies/tripadvisor_analysis.ipynb) — Classification and statistics on TripAdvisor reviews
- [`autoprofiling_biclass.ipynb`](case-studies/autoprofiling_biclass.ipynb) — Binary classification in author profiling

---

## 🛠️ Dependencies

- Python 3.x
- NLTK
- Scikit-learn
- NumPy, pandas, matplotlib
- TensorFlow or PyTorch for neural models

Install them via:

```bash
pip install -r requirements.txt
