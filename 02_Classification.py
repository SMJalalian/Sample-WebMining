import numpy as np
import pandas as pd
from Packages.My_NLP import KNN_Calculation
from Packages.Common import clearScreen

clearScreen()
# Load Trained twitter dataset ( with sentiment class and somw keyword frequency as feauter)
reference_dataset = pd.read_csv('Exports/TwitterSentimentDataset.csv')
#Load sample tweet for classification
sample_dataset = pd.read_csv('Exports/Sample_Tweet.csv')

#********************************************************************

features_of_reference_dataset = (reference_dataset.iloc[:, 2:]).head(200)
features_of_sample_dataset = (sample_dataset.iloc[:, 2:])
result = pd.DataFrame(columns=['Sentimental', 'cosinus_similarity'])

#********************************************************************

for sample_index, sample in features_of_sample_dataset.iterrows():
    for tweet_index, tweet in features_of_reference_dataset.iterrows():
        dotproduct_value = np.dot(tweet.to_numpy(), sample.to_numpy())
        norm_current_row = np.linalg.norm(tweet.to_numpy()) # calculate current row norm ( regarding to similarity formula )
        norm_sample = np.linalg.norm(sample.to_numpy()) # calculate sample data norm ( regarding to similarity formula )
        if (norm_current_row == 0 or norm_sample == 0): # prevent devide by zero exception
            cos = 0
        else:
            cos = dotproduct_value / (norm_current_row * norm_sample) # calculate cosinus similarity          
        result = result.append({'Sentimental': reference_dataset.at[tweet_index,"Sentimental"], 'cosinus_similarity': cos},ignore_index=True)
    
    print("*************************************** Base Information ***************************************")
    print("Text of Tweet: " , sample_dataset.at[sample_index,'Tweet'])
    print("Base Sentiment: " , sample_dataset.at[sample_index,'Sentimental'])
    print("================================")
    KNN_Calculation(result,5)  
    KNN_Calculation(result,20)
    KNN_Calculation(result,50)
    KNN_Calculation(result,100)
    result.drop(result.index, inplace=True)
    print("**********************************************************************************************")

#********************************************************************