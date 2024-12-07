import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
roberta = pd.read_csv('roberta_with_sentiment.csv')

# Pair-plot for to compare different Scores
sns.pairplot(data=roberta,
             vars=['roberta_pos', 'pos', 'roberta_neu', 'neu', 'roberta_neg', 'neg'],
             hue='Score',
             palette='tab10')
plt.show()

# Correlation matrix
corr_matrix = roberta[['roberta_pos', 'pos', 'roberta_neu', 'neu', 'roberta_neg', 'neg']].corr()

# Heatmap visualization
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f")
plt.title("Correlation Matrix of Sentiment Scores")
plt.show()

# Define the sentiment categories and corresponding columns
categories = [
    ("Positive 1 Star", 1, 'roberta_pos', 'pos'),
    ("Neutral 3 Star", 3, 'roberta_neu', 'neu'),
    ("Negative 5 Star", 5, 'roberta_neg', 'neg')
]

# Loop through the categories and print the results
for category, score, roberta_col, vader_col in categories:
    print(f"{category}: RoBERTa: ")
    print(roberta.query(f'Score == {score}').sort_values(roberta_col, ascending=False)['Text'].values[0])
    print(f"{category}: VADERS: ")
    print(roberta.query(f'Score == {score}').sort_values(vader_col, ascending=False)['Text'].values[0])