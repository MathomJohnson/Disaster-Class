import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


train = pd.read_csv('data/train.csv')
train.head()


# Function to tokenize text into words
def tokenize(text):
    # Remove non-alphanumeric characters and split into words
    words = re.findall(r'\b\w+\b', text.lower())  # Convert to lower case and extract words
    return [w for w in words if w not in stop_words]


# Separate the DataFrame based on the target
disaster_tweets = train[train['target'] == 1]
non_disaster_tweets = train[train['target'] == 0]


# Count occurrences of each unique word in disaster tweets
disaster_words = disaster_tweets['text'].apply(tokenize).explode()
disaster_word_counts = Counter(disaster_words)

# Count occurrences of each unique word in non-disaster tweets
non_disaster_words = non_disaster_tweets['text'].apply(tokenize).explode()
non_disaster_word_counts = Counter(non_disaster_words)

# Convert counts to DataFrames for better visualization
disaster_word_counts_df = pd.DataFrame(disaster_word_counts.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)
non_disaster_word_counts_df = pd.DataFrame(non_disaster_word_counts.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)

# Merge the two DataFrames on 'word'
merged_df = pd.merge(disaster_word_counts_df, non_disaster_word_counts_df, on='word', suffixes=('_disaster', '_non_disaster'))

# Calculate the ratio of disaster to non-disaster counts
merged_df['ratio'] = merged_df['count_disaster'] / merged_df['count_non_disaster']

similar_frequency_words = merged_df[abs(merged_df['ratio'] - 1) <= 0.25]

words_to_remove = similar_frequency_words['word'].tolist()
custom_stop_words = ['https']
words_to_remove.extend(custom_stop_words)

# Filtering out the words in both DataFrames
disaster_filtered = disaster_word_counts_df[~disaster_word_counts_df['word'].isin(words_to_remove)]
disaster_filtered = disaster_filtered[disaster_filtered['count'] > 3]

non_disaster_filtered = non_disaster_word_counts_df[~non_disaster_word_counts_df['word'].isin(words_to_remove)]
non_disaster_filtered = non_disaster_filtered[non_disaster_filtered['count'] > 3]


# Sort by count and select top N words
top_n = 30  # You can change this to any number you prefer
non_disaster_top = non_disaster_filtered.nlargest(top_n, 'count')
disaster_top = disaster_filtered.nlargest(top_n, 'count')

# Combine both datasets for plotting
non_disaster_top['category'] = 'Non-Disaster'
disaster_top['category'] = 'Disaster'

combined_top = pd.concat([non_disaster_top[['word', 'count', 'category']],
                          disaster_top[['word', 'count', 'category']]])

# Plotting
plt.figure(figsize=(20, 6))
sns.barplot(data=combined_top, x='word', y='count', hue='category')
plt.title('Top Words in Disaster vs. Non-Disaster Tweets')
plt.xlabel('Words')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Category')
plt.show()