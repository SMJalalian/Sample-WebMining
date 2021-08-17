# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from Packages.RecomHelper import get_Recommendations,cleaning_data,merge_All_Features

# Load movie dataset 
metadata = pd.read_csv('Datasets/ContenetBase_Recommenders_Movies_Metadata.csv')

#Create TF-IDF vectorize class and remove stopwords + NaNs
tfidf = TfidfVectorizer(stop_words='english')
metadata['overview'] = metadata['overview'].fillna('')
overview_tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Calculate cousin similarity matrix
overview_cosine_sim = linear_kernel(overview_tfidf_matrix, overview_tfidf_matrix)

#Create Moive index file due to search by index
indices = pd.Series(metadata.index, index=metadata['title'])

#Find recommended movies based on overview
get_Recommendations( metadata, indices, "Apollo 13", overview_cosine_sim)

#Cleaning data from some other columns 
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(cleaning_data)

#Merge all features (cast, keywords, directors, genres) to on column
metadata['allFeatures'] = metadata.apply(merge_All_Features, axis=1)

#Create vectorize data from all features
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['allFeatures'])

#Calculate cousin similarity matrix based on new features
allFeatures_cosine_sim = cosine_similarity(count_matrix, count_matrix)

#Create Moive index file due to search by index
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

#Find recommended movies based on cast, kewords, directors and genres
get_Recommendations(metadata, indices, 'Apollo 13', allFeatures_cosine_sim)