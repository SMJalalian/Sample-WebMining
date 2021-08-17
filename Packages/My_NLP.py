from itertools import groupby
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist
import re, string

from nltk.util import pr
from pandas.core.groupby.groupby import GroupBy

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
#*********************************************
def get_most_frequent_words(positive_tweet_tokens , negative_tweet_tokens , number_of_words = 10):
    # Create some empty list for future use ... 
    result = []
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    positive_list = []
    negative_list = []

    # create nltk stopwords + manuall stopwords
    stop_words = stopwords.words('english')
    stop_words += ["go","...","i'm", "get", "u", "day",'♛','》', "like", "follow" ] #add manually some words as a stopword

    for tokens in positive_tweet_tokens: # purify positive tokens 
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in negative_tweet_tokens: # purify negative tokens
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for item in positive_cleaned_tokens_list:
        positive_list += item # Collect all words in positive tweets
    freq_dist_pos = FreqDist(positive_list) # Count frequency of each unique word
    result += freq_dist_pos.most_common(number_of_words) # add most frequent word to the final result

    for item in negative_cleaned_tokens_list:
        negative_list += item # Collect all words in negative tweets
    freq_dist_neg = FreqDist(negative_list) # Count frequency of each unique word
    result += freq_dist_neg.most_common(number_of_words) # add most frequent word to the final result

    # retuen all positive & and negative words with frequency
    return result 
#*********************************************
def KNN_Calculation( inputDataframe, K = 5 ):
    Positive = 0
    Negative = 0
    sorted_data = inputDataframe.sort_values(by='cosinus_similarity',ascending=False)
    final_similarity_result = sorted_data.head(K)
    print("Calculation based on", K , "nearest neighbors")
    for index, item in final_similarity_result.iterrows():
        if item[0]=="Positive":
            Positive = Positive + 1
        else:
            Negative = Negative + 1
    print("Positive Class: ", Positive)
    print("Negative Class: ", Negative)
    if Positive > Negative:
        print('Pridicted Sentiment is: Positive')
    elif Negative > Positive:
        print('Pridicted Sentiment is: Negative')
    else:
        print('Pridicted Sentiment is Unknown !!!!')
    
    print("")
