# Sentiment Analysis and Text Visualization

This repository contains a comprehensive pipeline for analyzing and visualizing sentiments in text. It progresses from basic text cleaning and tokenization to advanced sentiment analysis using pre-trained models like VADER and RoBERTa. Visualizations are included to help better understand the results.

---

## Features

1. **Text Analysis**: 
   - Clean and tokenize raw text files.
   - Map words to emotions using a predefined `emotions.txt` file.
   - Generate emotion-based visualizations.

2. **Sentiment Analysis**:
   - Use the VADER model for sentiment scoring.
   - Use RoBERTa for a deep learning-based sentiment pipeline.

3. **Visualizations**:
   - Analyze the distribution of sentiments using bar graphs, pair plots, and heatmaps.
   - Compare VADER and RoBERTa sentiment scores.

4. **Interactive Pipeline**:
   - Utilize an interactive `pipeline.py` script to input custom text and get sentiment analysis on the fly.

---

## File Structure

### 1. TextAnalysis
- **`read.txt`**: The raw text file for initial analysis.
- **`emotions.txt`**: A mapping file to associate words with emotions.
- **`textimport.py`**:
  - Performs basic text cleaning, tokenization, and emotion extraction.
  - Generates a bar graph of emotions.
- **`graph.png`**: Emotion-based visualization from text analysis.

### 2. VADER-and-RoBERTa
- **`main.py`**: 
  - Uses VADERS to perform Ruled Based sentiment analysis on the dataset.
- **`robertatransformer.py`**: 
  - Uses RoBERTa to perform sentiment analysis on `vaders_with_sentiment.csv` dataset.
- **`visualize1.py` and `visualize2.py`**: 
  - Creates visualizations for VADER and RoBERTa sentiment scores.
- **`vaders_with_sentiment.csv`**: Output after VADER sentiment analysis.
- **`roberta_with_sentiment.csv`**: Output after RoBERTa sentiment analysis.

### 3. Pipleline-EXTRA
- **`pipeline.py`**: 
  - An interactive pipeline to input custom text and get sentiment results.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-project.git
   cd sentiment-analysis-project
