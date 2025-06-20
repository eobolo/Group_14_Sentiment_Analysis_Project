# -*- coding: utf-8 -*-
"""SVM MODEL_Amazon_Product_Sentiment_Analysis

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tyli4Ronw__tUeCI-F5Whu1JO3itzLJw

# Download the necessary libraries
"""

!pip install -U nltk
!python -m nltk.downloader all
!pip install -U gensim
!pip install keras
!pip install googletrans
!pip install langdetect
!pip install tensorflow==2.15.0
!pip install scikit-learn matplotlib

import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk
from langdetect import detect, LangDetectException
from googletrans import Translator
import time
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, Dense, GlobalAveragePooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam
import gensim.downloader as gensim_api
from gensim.models import Word2Vec
import pickle
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC

print("TensorFlow version:", tf.__version__)

# Base name for your files
base_name = 'dataset_amazon-sentiment-analysis'

# Create a list of file names
file_list = [f'{base_name}_{i}.csv' for i in range(1, 6)]

# Create an empty list to store DataFrames
df_list = []

# Loop through the file list, read each CSV, and append it to the list
for file in file_list:
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except FileNotFoundError:
        print(f"File not found: {file}")

# Concatenate the DataFrames in the list
if df_list:  # Check if the list is not empty
    merged_df = pd.concat(df_list, ignore_index=True)
else:
    print("No files were loaded.")

merged_df.shape

merged_df.columns

# Select the desired columns
filtered_df = merged_df[['reviewTitle', 'reviewDescription', 'ratingScore']]

# Display the first few rows of the modified DataFrame
filtered_df

filtered_df['ratingScore'].value_counts()

filtered_df[filtered_df['ratingScore'] == 4].tail(10)

filtered_df.loc[:, 'review'] = filtered_df['reviewTitle'].astype(str) + ' ' + filtered_df['reviewDescription'].astype(str)

# Create a new DataFrame with only the 'review' and 'ratingScore' columns
# This step is already creating a new DataFrame, so no .loc needed here
review_and_score_df = filtered_df[['review', 'ratingScore']].copy()

# Rename the 'ratingScore' column to 'score' for clarity
review_and_score_df.rename(columns={'ratingScore': 'score'}, inplace=True)

# Display the first few rows of the new DataFrame
review_and_score_df.head()

review_and_score_df.shape

# Create a new column 'sentiment' based on the 'score' column
# Assign 0 for scores 1, 2, and 3 (poor sentiment)
# Assign 1 for scores 4 and 5 (good sentiment)

review_and_score_df['sentiment'] = review_and_score_df['score'].apply(lambda x: 0 if x in [1, 2, 3] else 1)

# Display the first few rows of the DataFrame with the new 'sentiment' column
print(review_and_score_df.head())

# Check the distribution of the new 'sentiment' column
print("\nDistribution of the new 'sentiment' column:")
print(review_and_score_df['sentiment'].value_counts())

"""## Cleaning"""

# Initialize the translator
translator = Translator()

# --- Asynchronous Text Cleaning and Translation Function with Retry ---
async def clean_and_translate_text_async(text, remove_stopwords=False, max_retries=5, delay=1):
    """
    Asynchronously cleans text before translation, then translates with retries.
    """
    # 1. Handle empty or non-string input
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- Initial Cleaning (before translation) ---
    cleaned_text_initial = text.lower()
    cleaned_text_initial = cleaned_text_initial.translate(str.maketrans('', '', string.punctuation))
    cleaned_text_initial = re.sub(r'[^a-z\s]', '', cleaned_text_initial)
    cleaned_text_initial = re.sub(r'\s+', ' ', cleaned_text_initial).strip()

    if not cleaned_text_initial:
        return ""

    original_text_for_translation = cleaned_text_initial

    # 2. Detect language and translate to English if not with retry (asynchronous)
    translated_text = ""
    async with Translator() as translator: # Use async context manager
        for attempt in range(max_retries):
            try:
                # Langdetect is not async, so we use it directly
                lang = detect(original_text_for_translation)
                if lang != 'en':
                    # Translate to English (AWAIT the result)
                    translation_result = await translator.translate(original_text_for_translation, dest='en')
                    translated_text = translation_result.text
                else:
                    translated_text = original_text_for_translation
                break # Break out of retry loop if successful
            except LangDetectException:
                print(f"Language detection failed for text (after initial cleaning): {original_text_for_translation[:50]}...")
                return ""
            except Exception as e:
                print(f"Translation attempt {attempt + 1}/{max_retries} failed for text: {original_text_for_translation[:50]}... Error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay) # Use asyncio.sleep in async functions
                else:
                    print(f"Max retries reached. Translation failed for text: {original_text_for_translation[:50]}...")
                    return ""

    # --- Further Cleaning (after translation, if successful) ---
    text_for_further_cleaning = translated_text

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = text_for_further_cleaning.split()
        words = [w for w in words if w not in stop_words]
        text_for_further_cleaning = ' '.join(words)

    text_for_further_cleaning = text_for_further_cleaning.strip()

    return text_for_further_cleaning

# --- How to apply this to your DataFrame asynchronously ---

# Since pandas .apply() is not inherently asynchronous,
# you'll need to run the async function for each row using asyncio.

# Method 1: Using asyncio.gather() to run translations concurrently
async def translate_dataframe_async(df):
    # Create a list of coroutines (async function calls)
    coroutines_with_stopwords = [
        clean_and_translate_text_async(row['review'], remove_stopwords=False, max_retries=5)
        for index, row in df.iterrows()
    ]
    coroutines_without_stopwords = [
        clean_and_translate_text_async(row['review'], remove_stopwords=True, max_retries=5)
        for index, row in df.iterrows()
    ]

    # Run the coroutines concurrently
    results_with_stopwords = await asyncio.gather(*coroutines_with_stopwords)
    results_without_stopwords = await asyncio.gather(*coroutines_without_stopwords)

    # Assign results back to the DataFrame (or a new DataFrame)
    df['cleaned_review_translated_with_stopwords'] = results_with_stopwords
    df['cleaned_review_translated_without_stopwords'] = results_without_stopwords

    return df

# Run the async translation process
translated_df = await translate_dataframe_async(review_and_score_df.copy()) # If in an async cell
# Or
# translated_df = asyncio.run(translate_dataframe_async(review_and_score_df.copy())) # If in a sync cell

rows_to_drop = translated_df[
    (translated_df['cleaned_review_translated_with_stopwords'] == '') |
    (translated_df['cleaned_review_translated_without_stopwords'] == '')
].index

translated_df.drop(rows_to_drop, inplace=True)
translated_df.reset_index(drop=True, inplace=True)

# --- Extract Cleaned and Translated Sentences and Labels ---
# (Extract from translated_df)

sentences_translated_with_stopwords = translated_df['cleaned_review_translated_with_stopwords'].tolist()
sentences_translated_without_stopwords = translated_df['cleaned_review_translated_without_stopwords'].tolist()
labels = translated_df['sentiment'].tolist()

# Display and verification
print("First 5 cleaned and translated reviews with stopwords:")
print(sentences_translated_with_stopwords[:5])
print("\nFirst 5 cleaned and translated reviews without stopwords:")
print(sentences_translated_without_stopwords[:5])
print("\nFirst 5 labels:")
print(labels[:5])
print(f"\nShape of DataFrame after dropping rows with translation errors: {review_and_score_df.shape}")
print(f"\nNumber of reviews that became empty after cleaning and translation (with stopwords): {sentences_translated_with_stopwords.count('')}")
print(f"Number of reviews that became empty after cleaning and translation (without stopwords): {sentences_translated_without_stopwords.count('')}")

import os
# Define base directories for saving files
base_output_dir = 'cleaned_reviews_dataset'
stopwords_included_dir = os.path.join(base_output_dir, 'with_stopwords')
stopwords_excluded_dir = os.path.join(base_output_dir, 'without_stopwords')
labels_dir = os.path.join(base_output_dir, 'labels')

# Create directories if they don't exist
os.makedirs(stopwords_included_dir, exist_ok=True)
os.makedirs(stopwords_excluded_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Save each observation into a separate file
for i in range(len(sentences_translated_with_stopwords)):
    # Generate unique filename based on index
    filename = f'review_{i}.txt'
    label_filename = f'label_{i}.txt'

    # Save review with stopwords
    filepath_with_stopwords = os.path.join(stopwords_included_dir, filename)
    with open(filepath_with_stopwords, 'w', encoding='utf-8') as f:
        f.write(sentences_translated_with_stopwords[i])

    # Save review without stopwords
    filepath_without_stopwords = os.path.join(stopwords_excluded_dir, filename)
    with open(filepath_without_stopwords, 'w', encoding='utf-8') as f:
        f.write(sentences_translated_without_stopwords[i])

    # Save label
    filepath_label = os.path.join(labels_dir, label_filename)
    with open(filepath_label, 'w', encoding='utf-8') as f:
        f.write(str(labels[i])) # Convert label (integer) to string

print(f"Saved {len(sentences_translated_with_stopwords)} reviews and labels to individual files in '{base_output_dir}'")

# Define base directories where files are saved
base_output_dir = 'cleaned_reviews_dataset'
stopwords_included_dir = os.path.join(base_output_dir, 'with_stopwords')
stopwords_excluded_dir = os.path.join(base_output_dir, 'without_stopwords')
labels_dir = os.path.join(base_output_dir, 'labels')

# Lists to store the read data
loaded_sentences_with_stopwords = []
loaded_sentences_without_stopwords = []
loaded_labels = []

# Get list of files in the 'with_stopwords' directory (assuming consistent naming)
file_list = os.listdir(stopwords_included_dir)
file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0])) # Sort by the index in the filename

# Read each file and load the content
for filename in file_list:
    if filename.endswith('.txt'):
        # Extract the index from the filename
        try:
            index = int(filename.split('_')[1].split('.')[0])

            # Read review with stopwords
            filepath_with_stopwords = os.path.join(stopwords_included_dir, filename)
            with open(filepath_with_stopwords, 'r', encoding='utf-8') as f:
                loaded_sentences_with_stopwords.append(f.read())

            # Read review without stopwords (assuming same filename for corresponding review)
            filepath_without_stopwords = os.path.join(stopwords_excluded_dir, filename)
            with open(filepath_without_stopwords, 'r', encoding='utf-8') as f:
                loaded_sentences_without_stopwords.append(f.read())

            # Read label (assuming corresponding label file exists with same index)
            label_filename = f'label_{index}.txt'
            filepath_label = os.path.join(labels_dir, label_filename)
            with open(filepath_label, 'r', encoding='utf-8') as f:
                loaded_labels.append(int(f.read())) # Convert label back to integer

        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping file {filename} due to error: {e}")


# Display the first few loaded elements to verify
print("First 5 loaded reviews with stopwords:")
print(loaded_sentences_with_stopwords[:5])
print("\nFirst 5 loaded reviews without stopwords:")
print(loaded_sentences_without_stopwords[:5])
print("\nFirst 5 loaded labels:")
print(loaded_labels[:5])

# You can now create a new DataFrame from the loaded lists if needed
loaded_df = pd.DataFrame({
    'cleaned_review_translated_with_stopwords': loaded_sentences_with_stopwords,
    'cleaned_review_translated_without_stopwords': loaded_sentences_without_stopwords,
    'sentiment': loaded_labels
})

print("\nShape of the loaded DataFrame:")
print(loaded_df.shape)

loaded_df['sentiment'].value_counts()

sentences_translated_with_stopwords = loaded_df['cleaned_review_translated_with_stopwords'].tolist()
sentences_translated_without_stopwords = loaded_df['cleaned_review_translated_without_stopwords'].tolist()
labels = loaded_df['sentiment'].tolist()


# --- Word Tokenization (into lists) ---

# Tokenize the reviews with stopwords included
tokenized_sentences_with_stopwords = [nltk.word_tokenize(sentence) for sentence in sentences_translated_with_stopwords]

# Tokenize the reviews without stopwords included
tokenized_sentences_without_stopwords = [nltk.word_tokenize(sentence) for sentence in sentences_translated_without_stopwords]

# --- Get Vocabulary and Vocabulary Length ---

# Get the vocabulary (unique words) for reviews with stopwords
all_words_with_stopwords = [word for tokens in tokenized_sentences_with_stopwords for word in tokens]
vocabulary_with_stopwords = set(all_words_with_stopwords)
vocabulary_length_with_stopwords = len(vocabulary_with_stopwords)

# Get the vocabulary (unique words) for reviews without stopwords
all_words_without_stopwords = [word for tokens in tokenized_sentences_without_stopwords for word in tokens]
vocabulary_without_stopwords = set(all_words_without_stopwords)
vocabulary_length_without_stopwords = len(vocabulary_without_stopwords)

# --- Display Results ---

print("First 5 tokenized sentences with stopwords:")
print(tokenized_sentences_with_stopwords[:5])

print("\nFirst 5 tokenized sentences without stopwords:")
print(tokenized_sentences_without_stopwords[:5])

print(f"\nVocabulary length (unique words) with stopwords: {vocabulary_length_with_stopwords}")
print(f"Vocabulary length (unique words) without stopwords: {vocabulary_length_without_stopwords}")

# Optional: Display a few words from each vocabulary
print("\nFirst 10 words in vocabulary with stopwords:")
print(list(vocabulary_with_stopwords)[:10])

print("\nFirst 10 words in vocabulary without stopwords:")
print(list(vocabulary_without_stopwords)[:10])

# Assuming loaded_df is available from your preprocessing step
# Verify the DataFrame
print("First 5 rows of loaded DataFrame:")
print(loaded_df.head())
print("\nShape of loaded DataFrame:", loaded_df.shape)

# 1. Prepare features (X) and labels (y)
X_with_stopwords = loaded_df['cleaned_review_translated_with_stopwords']
X_without_stopwords = loaded_df['cleaned_review_translated_without_stopwords']
y = loaded_df['sentiment']

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train_with, X_test_with, y_train, y_test = train_test_split(
    X_with_stopwords, y, test_size=0.2, random_state=42, stratify=y
)
X_train_without, X_test_without, y_train, y_test = train_test_split(
    X_without_stopwords, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3)
)

# Fit and transform the training data (with stopwords)
X_train_with_tfidf = tfidf_vectorizer.fit_transform(X_train_with)
X_test_with_tfidf = tfidf_vectorizer.transform(X_test_with)

# 4. Train a Logistic Regression model
# Model for data with stopwords
model_with_stopwords = LogisticRegression(max_iter=1000, random_state=42)
model_with_stopwords.fit(X_train_with_tfidf, y_train)
y_pred_with = model_with_stopwords.predict(X_test_with_tfidf)

# 6. Evaluate the models
print("\n--- Model Performance (With Stopwords) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_with))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_with, target_names=['Bad', 'Good']))

# --- Visualize the Confusion Matrix ---

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_with)

# Create a ConfusionMatrixDisplay object
# display_labels are the labels you want to show on the plot (e.g., your class names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad', 'Good'])

# Plot the confusion matrix
# You can customize the appearance with parameters like cmap (colormap)
# include_values=True is the default, showing the counts in each cell [2]
disp.plot(cmap=plt.cm.Blues, values_format='d') # Use 'd' to display integers

# Add title
plt.title('Confusion Matrix (With Stopwords)')

# Show the plot
plt.show()

# Fit and transform the training data (without stopwords)
tfidf_vectorizer_without = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3)
)
X_train_without_tfidf = tfidf_vectorizer_without.fit_transform(X_train_without)
X_test_without_tfidf = tfidf_vectorizer_without.transform(X_test_without)

# 5. Make predictions
# Model for data without stopwords
model_without_stopwords = LogisticRegression(max_iter=1000, random_state=42)
model_without_stopwords.fit(X_train_without_tfidf, y_train)
y_pred_without = model_without_stopwords.predict(X_test_without_tfidf)

print("\n--- Model Performance (Without Stopwords) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_without))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_without, target_names=['Bad', 'Good']))

# --- Visualize the Confusion Matrix ---

# Calculate the confusion matrix
cm_without = confusion_matrix(y_test, y_pred_without)

# Create a ConfusionMatrixDisplay object
# display_labels are the labels you want to show on the plot (e.g., your class names)
disp_without = ConfusionMatrixDisplay(confusion_matrix=cm_without, display_labels=['Bad', 'Good'])

# Plot the confusion matrix
# You can customize the appearance with parameters like cmap (colormap)
# include_values=True is the default, showing the counts in each cell [2]
disp_without.plot(cmap=plt.cm.Blues, values_format='d') # Use 'd' to display integers

# Add title
plt.title('Confusion Matrix (Without Stopwords)')

# Show the plot
plt.show()

# 7. Optional: Feature Importance (Top TF-IDF features)
# Get feature names from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get coefficients from the model (with stopwords)
coef = model_with_stopwords.coef_[0]
top_positive_indices = np.argsort(coef)[-10:]  # Top 10 positive features
top_negative_indices = np.argsort(coef)[:10]   # Top 10 negative features

print("\n--- Top 10 Features for Positive Sentiment (With Stopwords) ---")
for idx in top_positive_indices:
    print(f"{feature_names[idx]}: {coef[idx]:.4f}")

print("\n--- Top 10 Features for Negative Sentiment (With Stopwords) ---")
for idx in top_negative_indices:
    print(f"{feature_names[idx]}: {coef[idx]:.4f}")

"""#**KERAS TOKENIZER AND WORD EMBEDDING**"""

print("First 5 rows of loaded DataFrame:")
print(loaded_df.head())

# --- Cell 2: Tokenization with TensorFlow.Keras Tokenizer ---
max_words = 5000  # Maximum vocabulary size
max_len = 100     # Maximum sequence length (adjust based on review length)

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(loaded_df['cleaned_review_translated_with_stopwords'])
sequences = tokenizer.texts_to_sequences(loaded_df['cleaned_review_translated_with_stopwords'])
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Prepare labels
y = loaded_df['sentiment'].values

# Save tokenizer for later use
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenization complete! Sequences shape:", X.shape)

# --- Cell 3: Train Custom TensorFlow.Keras Embedding ---
embedding_dim = 100  # Dimension of embedding vectors

# Split data for training the embedding
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a simple model to train the embedding
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(input_layer)
# Use GlobalAveragePooling1D to reduce 3D output to 2D
pooled = GlobalAveragePooling1D()(embedding_layer)
dense_layer = Dense(64, activation='relu')(pooled)
dense_layer = Dense(32, activation='relu')(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)
model = Model(inputs=input_layer, outputs=output_layer)

# define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
]

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy')

# Train the model to learn the embedding
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks=callbacks)

# Extract the trained embedding weights
keras_embedding_weights = model.layers[1].get_weights()[0]

# Prepare for visualization
word_index = tokenizer.word_index
words_to_visualize = 100  # Number of words to visualize
keras_vectors = []
keras_words = []
for word, i in word_index.items():
    if i < max_words and i > 0:  # Skip OOV and ensure within vocab
        keras_vectors.append(keras_embedding_weights[i])
        keras_words.append(word)
        if len(keras_vectors) >= words_to_visualize:
            break
keras_vectors = np.array(keras_vectors)

# Visualize
def plot_embeddings(vectors, labels, title, filename):
    if len(vectors) == 0:
        print(f"No vectors to visualize for {title}")
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=5 if len(vectors) > 10 else 2)
    vectors_2d = tsne.fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', alpha=0.5)
    for i, word in enumerate(labels):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9)
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(filename)
    plt.close()

plot_embeddings(keras_vectors, keras_words, 'Trained TensorFlow.Keras Embedding', 'keras_embedding_plot.png')
np.save('keras_embedding_weights.npy', keras_embedding_weights)

print("Custom TensorFlow.Keras embedding trained and visualized! Check keras_embedding_plot.png")

# --- Cell 4: Train Custom Word2Vec Embedding with Gensim ---
# Convert reviews to list of tokenized words for Word2Vec
tokenized_reviews = [review.split() for review in loaded_df['cleaned_review_translated_with_stopwords']]
word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=embedding_dim, window=5, min_count=5, workers=4, sg=1)

# Prepare for visualization
w2v_vectors = []
w2v_words = []
for word, i in word_index.items():
    if word in word2vec_model.wv and i < max_words:
        w2v_vectors.append(word2vec_model.wv[word])
        w2v_words.append(word)
        if len(w2v_vectors) >= words_to_visualize:
            break
w2v_vectors = np.array(w2v_vectors)

# Visualize and save
plot_embeddings(w2v_vectors, w2v_words, 'Custom Word2Vec Embedding', 'custom_word2vec_plot.png')
word2vec_model.save('custom_word2vec.model')
np.save('custom_word2vec_vectors.npy', w2v_vectors)

print("Custom Word2Vec embedding trained and visualized! Check custom_word2vec_plot.png")

# --- Cell 5: Load Pre-Trained Word2Vec Embedding ---
pretrained_word2vec = gensim_api.load('word2vec-google-news-300')
# Project to 100 dimensions for consistency
word2vec_dim = embedding_dim

# Prepare for visualization
pretrained_w2v_vectors = []
pretrained_w2v_words = []
for word, i in word_index.items():
    if word in pretrained_word2vec and i < max_words:
        vector = pretrained_word2vec[word][:embedding_dim]  # Project to 100 dims
        pretrained_w2v_vectors.append(vector)
        pretrained_w2v_words.append(word)
        if len(pretrained_w2v_vectors) >= words_to_visualize:
            break
pretrained_w2v_vectors = np.array(pretrained_w2v_vectors)

# Visualize and save
plot_embeddings(pretrained_w2v_vectors, pretrained_w2v_words, 'Pre-Trained Word2Vec Embedding', 'pretrained_word2vec_plot.png')
np.save('pretrained_word2vec_vectors.npy', pretrained_w2v_vectors)

print("Pre-Trained Word2Vec embedding loaded and visualized! Check pretrained_word2vec_plot.png")

# --- Cell 6: Load Pre-Trained GloVe Embedding ---
glove_path = '/content/glove.6B.100d.txt'  # Download from: https://nlp.stanford.edu/projects/glove/
if not os.path.exists(glove_path):
    print("Please download 'glove.6B.100d.txt' from https://nlp.stanford.edu/projects/glove/")
    print("Place it in the working directory and rerun this cell.")
else:
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Prepare for visualization
    glove_vectors = []
    glove_words = []
    for word, i in word_index.items():
        if word in glove_embeddings and i < max_words:
            glove_vectors.append(glove_embeddings[word])
            glove_words.append(word)
            if len(glove_vectors) >= words_to_visualize:
                break
    glove_vectors = np.array(glove_vectors)

    # Visualize and save
    plot_embeddings(glove_vectors, glove_words, 'Pre-Trained GloVe Embedding', 'glove_plot.png')
    np.save('glove_vectors.npy', glove_vectors)

    print("Pre-Trained GloVe embedding loaded and visualized! Check glove_plot.png")

"""#**LOADING BEST EMBEDDED LAYERS**"""

# Load tokenized data and labels
max_words = 5000
max_len = 100
embedding_dim = 100

if 'X' not in globals() or 'y' not in globals():
    if os.path.exists('tokenizer.pickle'):
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        print("Tokenizer not found! Please run previous tokenization code.")
        exit()
    sequences = tokenizer.texts_to_sequences(loaded_df['cleaned_review_translated_with_stopwords'])
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = loaded_df['sentiment'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load custom Word2Vec embedding matrix
if os.path.exists('custom_word2vec.model'):
    word2vec_model = Word2Vec.load('custom_word2vec.model')
    word_index = tokenizer.word_index
    embedding_matrix_w2v = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words and word in word2vec_model.wv:
            embedding_matrix_w2v[i] = word2vec_model.wv[word]
else:
    print("Custom Word2Vec model not found! Please run previous embedding generation code.")
    embedding_matrix_w2v = np.zeros((max_words, embedding_dim))

# Load GloVe embedding matrix
glove_path = '/content/glove.6B.100d.txt'
if os.path.exists(glove_path):
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    embedding_matrix_glove = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words and word in glove_embeddings:
            embedding_matrix_glove[i] = glove_embeddings[word]
else:
    print("GloVe file not found! Please download 'glove.6B.100d.txt' from https://nlp.stanford.edu/projects/glove/")
    embedding_matrix_glove = np.zeros((max_words, embedding_dim))

print("Data and embeddings loaded! X_train shape:", X_train.shape, "Embedding matrices shapes:", embedding_matrix_w2v.shape, embedding_matrix_glove.shape)

"""##**LSTM MODEL CREATION**"""

def create_lstm_model(embedding_matrix, max_words=5000, max_len=100, embedding_dim=100,
                      lstm_units=64, dense_units=32, trainable_embedding=False):
    """
    Create an LSTM model with a given embedding matrix (uncompiled).

    Parameters:
    - embedding_matrix: Pre-trained or custom embedding matrix (shape: max_words x embedding_dim)
    - max_words, max_len, embedding_dim: Embedding and sequence parameters
    - lstm_units: Number of LSTM units
    - dense_units: Number of units in dense layer
    - trainable_embedding: Whether the embedding layer is trainable

    Returns:
    - Uncompiled Keras model
    """
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim,
                               weights=[embedding_matrix], trainable=trainable_embedding)(input_layer)
    lstm_layer = LSTM(lstm_units, return_sequences=True)(embedding_layer)
    pooled = GlobalAveragePooling1D()(lstm_layer)
    dense_layer = Dense(dense_units, activation='relu')(pooled)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create, Compile, Train, and Evaluate LSTM with Custom Word2Vec
model_w2v = create_lstm_model(
    embedding_matrix=embedding_matrix_w2v,
    max_words=max_words, max_len=max_len, embedding_dim=embedding_dim,
    lstm_units=64, dense_units=32, trainable_embedding=False
)

# Compile model
optimizer = Nadam(learning_rate=0.00008)
model_w2v.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train model
history_w2v = model_w2v.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[reduce_lr, early_stopping], verbose=1
)

# Evaluate model
from sklearn.metrics import accuracy_score, classification_report
y_pred_w2v = (model_w2v.predict(X_test) > 0.5).astype(int)
print("\n--- LSTM with Custom Word2Vec Embedding ---")
print("Accuracy:", accuracy_score(y_test, y_pred_w2v))
print("Classification Report:")
print(classification_report(y_test, y_pred_w2v, target_names=['Bad', 'Good']))

# Save model
model_w2v.save('lstm_custom_word2vec.h5')

print("LSTM with Custom Word2Vec trained and evaluated! Model saved as lstm_custom_word2vec.h5")

# Create, Compile, Train, and Evaluate LSTM with Pre-Trained GloVe
# Create model
model_glove = create_lstm_model(
    embedding_matrix=embedding_matrix_glove,
    max_words=max_words, max_len=max_len, embedding_dim=embedding_dim,
    lstm_units=64, dense_units=32, trainable_embedding=False
)

# Compile model
optimizer = Adam(learning_rate=0.001)
model_glove.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Train model
history_glove = model_glove.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping, reduce_lr], verbose=1
)

# Evaluate model
y_pred_glove = (model_glove.predict(X_test) > 0.5).astype(int)
print("\n--- LSTM with Pre-Trained GloVe Embedding ---")
print("Accuracy:", accuracy_score(y_test, y_pred_glove))
print("Classification Report:")
print(classification_report(y_test, y_pred_glove, target_names=['Negative', 'Positive']))

# Save model
model_glove.save('lstm_glove.h5')

print("LSTM with Pre-Trained GloVe trained and evaluated! Model saved as lstm_glove.h5")

"""# **SVM MODEL**

I trained an SVM classifier using TF-IDF features generated from the review texts while retaining stopwords. The goal was to examine whether keeping common words like "not", "don’t", or "very"—which often carry important sentiment context—would improve model performance.

First, I initialized a TfidfVectorizer with a high-dimensional setting (max_features=8000) and a wide n-gram range (from unigrams to 5-grams). This allowed the model to capture longer word patterns that can provide richer context. I then applied this vectorizer to both the training and testing data.

Next, I used a GridSearchCV to perform hyperparameter tuning on the SVM model. I experimented with different values for the regularization parameter C, kernel types (linear and rbf), and gamma settings. This helped identify the optimal configuration for classifying the reviews.

After training, I evaluated the model using accuracy and a detailed classification report, which breaks down performance across the two sentiment classes: "Bad" and "Good". To better visualize the prediction performance, I plotted a confusion matrix showing how well the model distinguished between the two classes.
"""

# Fit and transform the training data (with stopwords)
tfidf_vectorizer_with = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 5)
)
X_train_with_tfidf = tfidf_vectorizer_with.fit_transform(X_train_with)
X_test_with_tfidf = tfidf_vectorizer_with.transform(X_test_with)

# Train SVM
print("\nTraining SVM Model with stopwords...")
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC(random_state=42)
svm_grid_search_with = GridSearchCV(svm, svm_param_grid, cv=3, n_jobs=-1, verbose=1)
svm_grid_search_with.fit(X_train_with_tfidf, y_train)
svm_pred_with = svm_grid_search_with.predict(X_test_with_tfidf)

print("\nSVM Results:")
print("Best parameters:", svm_grid_search_with.best_params_)
print("Accuracy:", accuracy_score(y_test, svm_pred_with))
print("\nClassification Report:")
print(classification_report(y_test, svm_pred_with, target_names=['Bad', 'Good']))

# Plot SVM confusion matrix
cm_svm = confusion_matrix(y_test, svm_pred_with)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Bad', 'Good'])
disp_svm.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix with stopwords')
plt.show()

"""I trained another SVM classifier, but this time after removing stopwords from the dataset. The aim was to test a more traditional approach, where common words like “the”, “is”, or “not” are excluded under the assumption that they don’t contribute much to the overall meaning or sentiment of the text.

To begin, I created a TfidfVectorizer with a reduced feature size (max_features=2000) and a slightly smaller n-gram range (1 to 3). Since stopwords were removed, the idea was to focus more on meaningful combinations of keywords without overfitting on less relevant patterns. I then transformed both training and test sets using this vectorizer.

For training, I used the same GridSearchCV strategy as before to find the best SVM configuration. Various combinations of C, kernel, and gamma parameters were tested through cross-validation to ensure a well-tuned model.

After training, the model’s accuracy and classification performance were evaluated and printed. A confusion matrix was also plotted to visually assess how well the model identified “Bad” vs. “Good” reviews.

This version helps highlight the differences in model behavior when stopwords are excluded versus when they are retained—contributing to the broader understanding of how preprocessing choices affect sentiment analysis outcomes.
"""

# Fit and transform the training data (without stopwords)
tfidf_vectorizer_without = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 3)
)
X_train_without_tfidf = tfidf_vectorizer_without.fit_transform(X_train_without)
X_test_without_tfidf = tfidf_vectorizer_without.transform(X_test_without)

# Train SVM
print("\nTraining SVM Model without stopwords...")
svm_param_grid_without = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm = SVC(random_state=42)
svm_grid_search_without = GridSearchCV(svm, svm_param_grid_without, cv=3, n_jobs=-1, verbose=1)
svm_grid_search_without.fit(X_train_without_tfidf, y_train)
svm_pred_without = svm_grid_search_without.predict(X_test_without_tfidf)

print("\nSVM Results:")
print("Best parameters:", svm_grid_search_without.best_params_)
print("Accuracy:", accuracy_score(y_test, svm_pred_without))
print("\nClassification Report:")
print(classification_report(y_test, svm_pred_without, target_names=['Bad', 'Good']))

# Plot SVM confusion matrix
cm_svm = confusion_matrix(y_test, svm_pred_without)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Bad', 'Good'])
disp_svm.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix without stopwords')
plt.show()

# Save the SVM model with stopwords
with open('svm_with_stopwords.h5', 'wb') as f:
    pickle.dump({
        'svm': svm_grid_search_with.best_estimator_,
        'vectorizer': tfidf_vectorizer_with,
        'parameters': svm_grid_search_with.best_params_
    }, f)

# Save the SVM model without stopwords
with open('svm_without_stopwords.h5', 'wb') as f:
    pickle.dump({
        'svm': svm_grid_search_without.best_estimator_,
        'vectorizer': tfidf_vectorizer_without,
        'parameters': svm_grid_search_without.best_params_
    }, f)