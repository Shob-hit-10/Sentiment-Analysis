import os
import pandas as pd

try:
    def load_imdb_data(data_dir):
        reviews = []
        sentiments = []

        for label in ['pos', 'neg']:
            folder = os.path.join(data_dir, label)
            for file in os.listdir(folder):
                with open(os.path.join(folder, file), encoding='utf-8') as f:
                    reviews.append(f.read())
                    sentiments.append('positive' if label == 'pos' else 'negative')

        return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# 🔁 Update this path if your dataset is in a different location
    train_path = r"C:\Users\Shobhit\Downloads\aclImdb_v1\aclImdb\train"

# Load and convert to DataFrame
    df_train = load_imdb_data(train_path)

# ✅ Save CSV to your Desktop
    output_path = r"C:\Users\Shobhit\OneDrive\Desktop\imdb_train_reviews.csv"
    df_train.to_csv(output_path, index=False)

    print(f"✅ CSV file saved at: {output_path}")
except Exception as e:
    print("❌ Error:", e)
