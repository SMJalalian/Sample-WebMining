# Import Required Libraries
import pandas as pd
from pandas.core.indexes.base import Index
from Packages.RecomHelper import weighted_rating
from Packages.Common import clearScreen

clearScreen()

# Load Simpe Movies Dataset
metadata = pd.read_csv('Datasets/Simple_Recommenders_Movies_Metadata.csv', low_memory=False)

# Calculate Average of all votes
ave_All_Votes = metadata['vote_average'].mean()

# Calculate the minimum number of votes that shoud be qualified
min_Required_Vote = metadata['vote_count'].quantile(0.90)

# Filter all qualified movies 
qualified_Movies = metadata.copy().loc[metadata['vote_count'] >= min_Required_Vote]

# Create new column (Score) to calculate ratings depend on theirs vote's count and sort them
qualified_Movies['score'] = qualified_Movies.apply(weighted_rating, args=(min_Required_Vote, ave_All_Votes), axis=1)
qualified_Movies = qualified_Movies.sort_values('score', ascending=False)

print((qualified_Movies[['score','title']].head(15)).to_string(index=False))

