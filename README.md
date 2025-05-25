# Sentiment Analysis of Airline Tweets using LSTM

## 📌 Project Overview

This project focuses on performing **Sentiment Analysis** on Twitter data related to airline services using **Natural Language Processing (NLP)** and **Deep Learning** techniques. The goal is to classify tweets into three sentiment categories: **positive**, **negative**, and **neutral** using an LSTM-based neural network model implemented in **PyTorch**.

---

## 🧠 Technologies & Libraries Used

- **Python 3**
- **Pandas**, **NumPy**
- **Regular Expressions (re)** for text cleaning
- **scikit-learn** (LabelEncoder, train_test_split)
- **Keras** (Tokenizer, pad_sequences)
- **PyTorch** (torch, nn, optim, DataLoader)

---

## 🗂 Dataset

- Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Contains ~14,000 tweets labeled as `positive`, `neutral`, or `negative`.

---

## 🧹 Data Preprocessing

1. Selected only relevant columns: `text` and `airline_sentiment`.
2. Cleaned the text by removing:
   - URLs
   - Mentions (@user)
   - Hashtags
   - Special characters and digits
3. Tokenized and padded the cleaned text using Keras.
4. Encoded sentiment labels to numerical form using `LabelEncoder`.

---

## 🔍 Model Architecture

- **Embedding Layer:** Converts input words into dense vector representations.
- **LSTM Layer:** Captures sequential patterns and long-term dependencies in text.
- **Dropout Layer:** Prevents overfitting.
- **Fully Connected Layer:** Outputs class probabilities for 3 sentiment categories.

```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=10):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        return self.fc(out)
```

🏋️ Model Training
Loss Function: CrossEntropyLoss

Optimizer: Adam

Batch Size: 64

Epochs: 10

The model was trained and validated on an 80-20 split of the dataset.


📈 Results
Test Accuracy: 64.52%

Performance is promising given the simplicity of the model and no pre-trained embeddings.

Accuracy can be improved using techniques such as:

Pretrained embeddings (GloVe, Word2Vec)

Bidirectional LSTM

Attention mechanisms

✅ Future Improvements
Use pretrained language models like BERT for better contextual understanding.

Hyperparameter tuning for better accuracy.

Implement a web interface using Flask/Streamlit for live sentiment prediction.

🙋‍♂️ Author
Jerin Abraham
B.Tech in Computer Engineering (Big Data Analytics)
Passionate about AI, NLP, and Deep Learning
