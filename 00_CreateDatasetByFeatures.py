from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples
from pandas.core.frame import DataFrame
from Packages.My_NLP import get_most_frequent_words

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json') # Load tokenized words from 5000 positive tweets
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json') # Load tokenized words from 5000 negative tweets

#Create 10 most frequent wrods from each positive and negative tweets
most_repetitive_words = get_most_frequent_words(positive_tweet_tokens, negative_tweet_tokens, 10)

#*******************************************

positive_tweets = twitter_samples.strings('positive_tweets.json')  # Load 5000 raw positive tweet from library as a reference
dataset_positive = DataFrame(positive_tweets,columns=['Tweet']) # Create a Dataframe by just tweet's text
dataset_positive["Sentimental"] = "Positive" # add positive ( as a sentiment ) to the dataframe's next column

negative_tweets = twitter_samples.strings('negative_tweets.json') # Load 5000 raw negative tweet from library as a reference
dataset_negative = DataFrame(negative_tweets,columns=['Tweet']) # Create a Dataframe by just tweet's text
dataset_negative["Sentimental"] = "Negative" # add negative ( as a sentiment ) to the dataframe's next column

#Create complete dataframe from both negative abd positive tweets + sentiment column
final_dataset = dataset_positive.append(dataset_negative,ignore_index=True)

#*******************************************

for value in most_repetitive_words :
    final_dataset[value[0]]= 0 # add Zero (0) to all cells in final dataset ( dataset initialized by zero value )

tkz = TweetTokenizer()
for index, row in final_dataset.iterrows(): # read all final dataset ( row by row )
    for col in most_repetitive_words: # for each keyword 
        TempTokenize = tkz.tokenize(row['Tweet']) # tokenize text of tweet ( in each row )
        final_dataset.at[index,col[0]] = TempTokenize.count(col[0]) # calculate frequency of keyword in tweet text
        
#*******************************************
# Export trained dataset with classifier ( Positive/Negative ) and all kewords frequency ( Features )
final_dataset = final_dataset.sample(frac=1).reset_index(drop=True)
sample_dataset = final_dataset.iloc[:20, :] #Generate sample data
trained_dataset = final_dataset.iloc[20:, :] #Generate rest data as reference file
sample_dataset.to_csv("Exports/Sample_Tweet.csv",index=False) 
trained_dataset.to_csv("Exports/TwitterSentimentDataset.csv",index=False) 
#*******************************************