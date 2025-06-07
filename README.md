# Group 14 - Sentiment Analysis Project

## Project Overview

This project aims to build a **text classification system for sentiment analysis**, comparing the performance of **traditional machine learning** and **deep learning models**. We explored the **Amazon reviews dataset**, conducted extensive **EDA and preprocessing**, and implemented both **Logistics Regression** and **LSTM** models for sentiment classification. Our objective was to identify the best-performing approach in classifying text sentiment.

## Dataset

- **Source:** Amazon Reviews  
- **Description:** Contains over 2,000 reviews labeled as either *positive* or *negative*.  
- **Classes:** Binary sentiment classification (positive, negative)


## Repository Structure

The repository is organized into the following directories and files for clarity and reproducibility:

- datasets/: Folder for input data.
  - dataset_amazon-sentiment-analysis_1.csv.
  - dataset_amazon-sentiment-analysis_2.csv.
  - dataset_amazon-sentiment-analysis_3.csv.
  - dataset_amazon-sentiment-analysis_4.csv.
  - dataset_amazon-sentiment-analysis_5.csv.
    
- models/: Stored model predictions, and submission files.
  - Deep_Learning_LSTM_Model.h5
  - lstm_custom_word2vec_model7.h5
  - lstm_custom_word2vec_model7b.h5
  - lstm_custom_word2vec_model7c.h5
  - lstm_custom_word2vec_model7d.h5
  - svm_with_stopwords.h5
  - svm_without_stopwords.h5
    
- notebooks
  - Group14_Amazon_Product_Sentiment_Analysis_Cleaning.ipynb
  - Group14_Amazon_Product_Sentiment_Analysis_Deep_Learning_Model.ipynb
  - Group14_Amazon_Product_Sentiment_Analysis_SVM_Model.ipynb
  - group14_amazon_product_sentiment_analysis_cleaning.py
  - svm_model_amazon_product_sentiment_analysis.py
    
- README.md: This file, providing an overview and instructions.
  
- requirements.txt: Lists Python dependencies for reproducibility.

## Data Sourcing

- **Source**: Amazon product reviews scraped using [Apify](https://apify.com/)

The files are placed in the datasets/ directory as follows:
  - dataset_amazon-sentiment-analysis_1.csv.
  - dataset_amazon-sentiment-analysis_2.csv.
  - dataset_amazon-sentiment-analysis_3.csv.
  - dataset_amazon-sentiment-analysis_4.csv.
  - dataset_amazon-sentiment-analysis_5.csv.
  
- **Features**:
  
  - `reviewTitle`: Brief summary of the review
    
  - `reviewDescription`: Full review text
    
  - `ratingScore`: Rating (1–5 stars)
    
- **Sentiment Labels**:
  
  - Ratings 1–3 → Negative (0)
    
  - Ratings 4–5 → Positive (1)

## Models Implemented

### Traditional Machine Learning

- **Logistic Regression**
  
- **Support Vector Machine (SVM)**
  
  - GridSearchCV for tuning `C`, `kernel`, `gamma`
    
- **Features**: TF-IDF vectors with various n-gram ranges and feature sizes (1000–8000)

### Deep Learning

- **LSTM Network**:
  
  - Embedding layer (custom/trainable/pretrained)
    
  - LSTM (64 units)
    
  - GlobalAveragePooling
    
  - Dense layer (ReLU)
    
  - Sigmoid output
    
- **Optimizers**: Adam, Nadam
  
- **Embeddings**:
  
  - Keras trainable embedding
    
  - Custom Word2Vec (Gensim)
    
  - Pre-trained Word2Vec (Google News)
    
  - Pre-trained GloVe (100d)
    
- **Loss Function**: Binary cross-entropy
  
- **Regularization**: EarlyStopping and Learning Rate Reduction

##  Setup Instructions

Follow these steps to set up and run the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/eobolo/Group_14_Sentiment_Analysis_Project.git
cd Group_14_Sentiment_Analysis_Project

# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

### install dependecies
pip install -r requirements.txt
```

## Additional Notes

- The scripts assume the data files are in the datasets/ directory. Adjust file paths in the notebook if your setup differs.
- The project uses TensorFlow for LSTM implementation; ensure your environment supports it.
