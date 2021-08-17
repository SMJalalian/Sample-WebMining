#Load Libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from Packages.RecomHelper import get_movie_recommendation

#Load Datasets
movies = pd.read_csv("Datasets/movies.csv")
ratings = pd.read_csv("Datasets/ratings.csv")

#Create New dataframe based on Pivot
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)

#Count vote and users numbers
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

#Filter low rated move and users
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset = final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

#Create spares matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

#Calculate collaborative filtering base on 20 KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

print(get_movie_recommendation(final_dataset, csr_data, knn, movies, 10, "Jumanji"))