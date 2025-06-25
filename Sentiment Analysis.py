# sentiment_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords

# Only download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# 1. Load dataset
df = pd.read_csv(r"C:\Users\Shobhit\OneDrive\Desktop\imdb_train_reviews.csv") # Load your CSV file
print("Sample data:")
print(df.head())

# 2. Basic cleaning
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove numbers and punctuation
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

# 3. Convert sentiment to binary (positive=1, negative=0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42)

# 5. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 6. Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 9. WordCloud (Optional Visualization)
pos_words = ' '.join(df[df['sentiment'] == 1]['clean_review'])
neg_words = ' '.join(df[df['sentiment'] == 0]['clean_review'])

wc = WordCloud(width=800, height=400, background_color='white').generate(pos_words)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Words WordCloud")
plt.show()

wc = WordCloud(width=800, height=400, background_color='white').generate(neg_words)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Words WordCloud")
plt.show()
