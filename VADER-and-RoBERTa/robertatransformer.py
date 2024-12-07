import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer and RoBERTa model/tokenizer
sia = SentimentIntensityAnalyzer()
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

vad = pd.read_csv('vaders_with_sentiment.csv')

# Define the function for RoBERTa polarity scores
def polarity_scores_roberta(example):
    tokenized_text = tokenizer(example, return_tensors='pt')
    output = model(**tokenized_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# Initialize a dictionary to store the results
res = {}

# Process each row in the dataframe
for i, row in tqdm(vad.iterrows(), total=len(vad)):
    try:
        text = row['Text']
        unique_id = row['uniqueid']

        # Perform RoBERTa sentiment analysis
        roberta_result = polarity_scores_roberta(text)

        # Store the results in the dictionary
        res[unique_id] = roberta_result
    except RuntimeError:
        print(f'Broke for uniqueid {unique_id}')

# Convert the results dictionary to a DataFrame through Transform function 'T'
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'uniqueid'})
results_df = results_df.merge(vad, how='left')

# Save the new DataFrame to a CSV file if desired
results_df.to_csv('roberta_with_sentiment.csv', index=False)

# Display a sample of the new DataFrame
results_df.head()