# Importing Necessary Libraries
import string
import pandas as pd

from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('Reviews.csv')
print("Shape=", df.shape)

# Use only the first 500 rows
df = df.head(500)

# Cleaning Text = Converting Text to LowerCase + Removing Punctuation
def preprocess_text(text):
      # Convert text to lowercase
      text = text.lower()
      # Remove punctuation
      text = text.translate(str.maketrans('', '', string.punctuation))
      # Tokenize text
      tokenized_text = word_tokenize(text, 'english')
      # Remove stop words
      stop_words = stopwords.words('english')
      clean_tokens = [word for word in tokenized_text if word not in stop_words]
      # Join tokens back into a single string
      return ' '.join(clean_tokens)


# Apply the preprocessing function to the Text column
df['Cleaned Text'] = df['Text'].apply(preprocess_text)

# SentimentIntensityAnalyzer instance
sia = SentimentIntensityAnalyzer()

# Initialize a dictionary to hold the sentiment scores
res = {}

# Iterate over the rows of the dataframe and calculate sentiment scores
for i, row in tqdm(df.iterrows(), total=len(df)):
      text = row['Text']
      myid = row['Id']
      res[myid] = sia.polarity_scores(text)

# Convert the dictionary into a DataFrame
after_vad = pd.DataFrame(res).T
after_vad = after_vad.reset_index().rename(columns={'index': 'Id'})


# Define a function to classify sentiment
def classify_sentiment(row):
      pos = row['pos']
      neg = row['neg']

      if pos > neg:
            return 'Positive'
      elif pos < neg:
            return 'Negative'
      else:
            return 'Neutral'


# Add a new column for Sentiment classification (Positive/Negative/Neutral)
after_vad['Sentiment'] = after_vad.apply(classify_sentiment, axis=1)

# Merge the VADERS DataFrame with the original DataFrame to retain all the original columns
after_vad= after_vad.merge(df, how='left', on='Id')

# Renaming Id Column
after_vad.rename(columns={'Id': 'uniqueid'}, inplace=True)

# Display the first few rows of the new VADERS dataset
print(after_vad.head())

after_vad.to_csv('vaders_with_sentiment.csv', index=False)