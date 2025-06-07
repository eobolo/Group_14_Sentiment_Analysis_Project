# 📦 Sentiment Analysis of Amazon Product Reviews  
*Group 14 - Natural Language Processing Assignment*

This project explores sentiment analysis on Amazon product reviews using both traditional machine learning and deep learning techniques. The goal is to classify reviews into **positive** or **negative** sentiment classes.

📂 GitHub Repository: [Group_14_Sentiment_Analysis_Project](https://github.com/eobolo/Group_14_Sentiment_Analysis_Project)

---

## ✨ Project Highlights

- Data collected using **Apify** web scraping tools from Amazon.
- Language detection and translation for non-English reviews.
- Preprocessing includes text cleaning, tokenization, stopword management.
- Comparison between **Logistic Regression**, **SVM**, and **LSTM** models.
- Feature engineering via **TF-IDF**, **trainable embeddings**, **Word2Vec**, and **GloVe**.
- Analysis of the effect of **stopwords**, **class imbalance**, and **embedding strategies**.

---

## 🗂️ Repository Structure

The GitHub repository is well-structured with a `README.md` explaining the project and instructions for reproducing results. Files are logically organized (e.g., separate folders for `scripts/`, `data/`, and `outputs/`) to ensure clarity and reproducibility.


<pre> <code> Group_14_Sentiment_Analysis_Project/
  ├── datasets/ 
  ├── notebooks/ 
  ├── models/ 
  ├── outputs/
  ├── report/
  ├── requirements.txt 
  └── README.md
</pre>
---

## 📊 Dataset Overview

- **Source**: Scraped from Amazon via Apify.
- **Features**:
  - `reviewTitle`
  - `reviewDescription`
  - `ratingScore` (1 to 5)
- **Labeling Strategy**:
  - 1–3 stars → **Negative (0)**
  - 4–5 stars → **Positive (1)**
- Reviews were translated to English where necessary and cleaned.

---

## 🛠️ Methodology

### 🔡 Preprocessing

- **Language Detection**: `langdetect`
- **Translation**: `googletrans`
- **Text Cleaning**: Lowercasing, punctuation removal, non-alphabetic filtering
- **Stopwords**: Experiments with and without removal (NLTK stopword list)
- **Storage**: Text and labels stored in separate files for flexibility

### 🧪 Feature Engineering

- **TF-IDF**: Used for traditional models (Logistic Regression, SVM)
- **Tokenization**: `Tokenizer` from TensorFlow
- **Embeddings**:
  - Trainable Keras Embedding
  - Custom Word2Vec (Gensim)
  - Pretrained Word2Vec (Google News)
  - Pretrained GloVe (100d)

---

## 🤖 Models & Configurations

### Traditional Models
- **Logistic Regression**: Simple and interpretable
- **SVM**: Grid-searched over kernel (linear, RBF), C, gamma

### Deep Learning Model
- **LSTM Network**:
  - Input (100-token sequences)
  - Embedding Layer
  - LSTM (64 units)
  - GlobalAveragePooling
  - Dense (32 units, ReLU)
  - Output (Sigmoid for binary classification)
- **Optimizers**: Adam, Nadam
- **Loss Function**: Binary Cross-Entropy
- **Training Strategies**: EarlyStopping, Learning Rate Decay

---

## 📈 Results & Observations

### Traditional Models
- Logistic Regression and SVM were surprisingly competitive
- Stopwords improved model performance by retaining important words (e.g., "not", "don't")

### LSTM Results
- Performance was promising but affected by **class imbalance**
- Word embeddings impacted performance, with pretrained vectors showing more stable results
- Models were prone to overfitting; early stopping helped

### Key Metrics (examples):
- Accuracy: 85–88%
- F1-Score: Higher for positive class; lower recall for negatives

---

## 🔍 Challenges & Limitations

- **Severe Class Imbalance**: Skewed toward positive reviews
- **Machine Translation Limitations**: May introduce errors
- **Model Scope**: Only basic LSTM explored (no BERT or GRU)
- **Domain Specificity**: Results may not generalize beyond Amazon

---

## 📌 Recommendations

1. **Balance the dataset** with synthetic sampling or more negative reviews.
2. Try transformer-based models like **BERT** or **RoBERTa**.
3. Explore **ensemble techniques** combining ML and DL approaches.
4. Test generalizability on reviews from **other domains**.
5. Build a **real-time review analysis tool**.

---

## 👥 Team Contributions

| Team Member            | Contribution                                                                 |
|------------------------|------------------------------------------------------------------------------|
| **Emmanuel Obolo**     | Data scraping, cleaning, preprocessing, Logistic Regression model            |
| **Jordan Nguepi**      | Hyperparameter tuning, metric evaluation for ML models                      |
| **Jamillah Ssozi**     | SVM implementation and documentation                                         |
| **Justice Chukwuonye** | LSTM model development and evaluation                                        |

---

## 📄 Report & References

- 📑 [View Full PDF Report](https://docs.google.com/document/d/109xSSswcFlkdN51cKMkEjtr58hoZsCwgBvQUoW_207U)
- 📊 [Experiment Tables](https://docs.google.com/spreadsheets/d/1YLojFlst-nGRykdGIQNkC7_fWPJjrQI9AVLKZh4nTvY/edit?usp=sharing)

---

## 🧾 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
