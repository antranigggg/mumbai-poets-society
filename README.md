Mumbai Poets Society Poetry Analyzer

Note: While this project is facing compatibility issues with Streamlit, cloning the repository and running it locally with the necessary requirements installed will allow you to use this program.
The develepor is still working towards deployment and shall achieve it soon.

Overview

The Mumbai Poets Society Poetry Analyzer is a deep learning-powered web application that analyzes poetry to uncover emotional, thematic, and literary insights. Built as part of my initiative with the Mumbai Poets Society, this project applies state-of-the-art natural language processing (NLP) techniques to bridge poetry and artificial intelligence, offering poets and enthusiasts a tool to explore the emotional and narrative essence of their work. As a freshman at VIT Vellore, I developed this project to demonstrate my self-taught skills in deep learning and my passion for interdisciplinary applications of AI.


Features

Emotion Analysis: Uses the j-hartmann/emotion-english-distilroberta-base transformer model to classify emotions (e.g., happiness, sadness, anger) in poems, with custom post-processing to enhance synergy (e.g., happy-hopeful) and reduce conflicts (e.g., happy-sadness).

Theme Detection: Combines SentenceTransformers (all-distilroberta-v1) with TF-IDF and semantic similarity to identify themes like love, hope, or loss, prioritizing light themes for nuanced analysis.

Literary Device Detection: Identifies figurative language (e.g., metaphors, similes, alliteration) using SpaCy dependency parsing and regex, paired with lexical diversity and readability metrics.

Playlist Generation: Matches poems to songs based on emotional and lyrical similarity using cosine similarity on transformer embeddings, creating personalized poetic playlists.

Narrative Archetypes: Classifies poems into archetypes (e.g., Hero’s Journey, Fall from Grace) based on weighted scoring of emotions and themes, with a badge system (e.g., The Visionary) for user engagement.

Interactive Visualizations: Displays results via Plotly bar and radar charts, with WordClouds for word frequencies, enhancing user understanding of model outputs.

Professional Reports: Generates LaTeX and text-based reports summarizing analysis, suitable for academic or literary use.


Tech Stack

Deep Learning: Hugging Face Transformers (DistilRoBERTa), SentenceTransformers

NLP Libraries: NLTK, SpaCy, TextBlob, VADER, NRCLex

Web Framework: Streamlit (deployed on Streamlit Cloud

Visualization: Plotly, WordCloud

Data Management: JSON for persistent song database

Others: Python, LaTeX, Git


Key Technical Achievements

Transformer Integration: Leveraged DistilRoBERTa for emotion classification and SentenceTransformers for semantic embeddings, achieving high-confidence emotion detection (threshold >0.7) and precise playlist matching.


Interpretability: Enhanced model transparency with detailed explanations for predictions, using WordNet and dependency parsing to contextualize results for non-technical users.

Optimization: Utilized Streamlit’s @st.cache_resource to cache transformer models, reducing inference time for real-time analysis.


Challenges Overcome

Emotion Analysis: Improved song lyric analysis by tuning emotion vectors to avoid uniform distributions, increasing semantic similarity weight by 20%.

LaTeX Generation: Resolved syntax errors in report generation by preprocessing poem text to escape special characters, ensuring robust outputs.


Future Improvements

Add a speech-to-text service that will allow greater usability.

Fine-tune DistilRoBERTa on a poetry-specific dataset to improve emotion detection accuracy.

Add evaluation metrics (e.g., accuracy, F1 score) using a labeled poetry dataset for research rigor.

Implement attention visualization with bertviz to showcase transformer focus on key poetic words.

Modularize code into separate modules (e.g., emotion_analyzer.py) for better maintainability.




Access the live demo: [Streamlit Cloud Link] (if hosted)

Sample Output



# mumbai-poets-society
