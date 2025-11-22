# Multi-Modal AI Chatbot for Movies & TV Series

This repository contains the source code and documentation for a multi-modal AI chatbot that answers questions about movies and TV series. The system combines pattern-based conversation, semantic retrieval, logical reasoning, voice interaction, and image classification of movie posters into a single integrated platform.

> 📄 Full coursework documentation with screenshots and explanations:  
> `docs/AI_Chatbot_Project_Report.pdf`  
>
> 🎥 Demo video (15 min): https://youtu.be/Wdl4M2eGC3Q

---

## Project Goals

The chatbot is designed to:

- Hold topic-specific conversations about movies and TV series
- Retrieve answers using both pattern-matching and similarity search
- Reason over a knowledge base using first-order logic (FOL)
- Support voice input and spoken responses
- Classify movie posters with a CNN and explain predictions using GradCAM

---

## Main Features

### 1. Conversational Capabilities (Task A)

- **AIML Pattern-Matching**  
  An AIML file (`data/mybot-basic.xml`) defines conversation patterns for common queries such as release dates, directors, and genres.

- **TF-IDF + Cosine Similarity Fallback**  
  When a query doesn’t match any AIML pattern, the system:
  - Lemmatizes the text
  - Computes TF-IDF vectors
  - Uses cosine similarity against a CSV of Q&A pairs (`data/qa_pairs.csv`)  
  to retrieve the closest answer.

- **Voice Interaction (Extra)**  
  Using `SpeechRecognition` and `pyttsx3`, the chatbot can:
  - Capture spoken questions
  - Convert them to text
  - Reply with synthesized speech

Core files:
- `src/chatbot/main.py`
- `src/chatbot/similarity_bot.py`
- `src/chatbot/voice_interface.py`
- `data/qa_pairs.csv`
- `data/mybot-basic.xml`

---

### 2. Rule-Based Logical Reasoning (Task B)

- **First-Order Logic with NLTK**  
  A movie knowledge base is stored in FOL form (e.g. `Director(Christopher_Nolan, Inception)`), using NLTK’s logic tools.

- **Two Input Patterns**  
  - `"I know that X is Y"` → add a fact to the KB after checking for contradictions  
  - `"Check that X is Y"` → verify the fact; respond with `Correct`, `Incorrect`, or `Unknown`

- **Fuzzy Fact Checker (Extra)**  
  Uses Python’s `difflib` to handle near-miss inputs and returns a confidence score for approximate matches.

Core files:
- `src/chatbot/logic_reasoning.py`
- `data/knowledge_base.txt`

---

### 3. Image Classification & GradCAM (Task C)

- **CNN Movie Poster Classifier**  
  A convolutional neural network (`models/my_cnn_model.h5`) is trained with TensorFlow/Keras on movie posters.  
  It predicts the movie and returns a confidence score.

- **GradCAM Visualiser (Extra)**  
  GradCAM is used to generate a heatmap highlighting image regions that most influenced the prediction. Example outputs are saved as:
  - `examples/gradcam_output.jpg`
  - `examples/gradcam_comparison.jpg`

- **Input Options**  
  - Image URL  
  - Local file selection

Core files:
- `src/chatbot/image_classifier.py`
- `src/chatbot/gradcam_utils.py`
- `models/my_cnn_model.h5`
- `models/class_indices.pkl`
