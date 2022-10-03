
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


### https://github.com/rposhala/Recommender-System-on-MovieLens-dataset/blob/main/Item_based_Collaborative_Recommender_System_using_KNN.ipynb

overall_stats = pd.read_csv('data/u.info', header=None)
#print("Details of users, items and ratings involved in the loaded movielens dataset: ",list(overall_stats[0]))


## same item id is same as movie id, item id column is renamed as movie id
column_names1 = ['user id','movie id','rating','timestamp']
dataset = pd.read_csv('data/u.data', sep='\t',header=None,names=column_names1)
#print(dataset.head()) 

#print(len(dataset), max(dataset['movie id']),min(dataset['movie id']))


d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
items_dataset = pd.read_csv('data/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
#print(items_dataset)


movie_dataset = items_dataset[['movie id','movie title']]
#print(movie_dataset.head())

## looking at length of original items_dataset and length of unique combination of rows in items_dataset after removing movie id column
#print(len(items_dataset.groupby(by=column_names2[1:])),len(items_dataset))


merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movie id')
merged_dataset[(merged_dataset['movie title'] == 'Chasing Amy (1997)') & (merged_dataset['user id'] == 894)]
refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})
#print(refined_dataset.head())

# num_users = len(refined_dataset.rating.unique())
# num_items = len(refined_dataset.movieId.unique())
num_users = len(refined_dataset['user id'].value_counts())
num_items = len(refined_dataset['movie title'].value_counts())
#print('Unique number of users in the dataset: {}'.format(num_users))
#print('Unique number of movies in the dataset: {}'.format(num_items))

rating_count_df = pd.DataFrame(refined_dataset.groupby(['rating']).size(), columns=['count'])
#print(rating_count_df)

# ax = rating_count_df.reset_index().rename(columns={'index': 'rating score'}).plot('rating','count', 'bar',
#     figsize=(12, 8),
#     title='Count for Each Rating Score',
#     fontsize=12)

# ax.set_xlabel("movie rating score")
# ax.set_ylabel("number of ratings")
# plt.show()
# print(ax)

total_count = num_items * num_users
zero_count = total_count-refined_dataset.shape[0]
#print(zero_count)

# append counts of zero rating to df_ratings_cnt
rating_count_df = rating_count_df.append(
    pd.DataFrame({'count': zero_count}, index=[0.0]),
    verify_integrity=True,
).sort_index()
#print(rating_count_df)

# add log count
rating_count_df['log_count'] = np.log(rating_count_df['count'])
rating_count_df

rating_count_df = rating_count_df.reset_index().rename(columns={'index': 'rating score'})
rating_count_df

# ax = rating_count_df.plot('rating score', 'log_count', 'bar', figsize=(12, 8),
#     title='Count for Each Rating Score (in Log Scale)',
#     logy=True,
#     fontsize=12,)

# ax.set_xlabel("movie rating score")
# ax.set_ylabel("number of ratings")

refined_dataset.head()

# get rating frequency
movies_count_df = pd.DataFrame(refined_dataset.groupby('movie title').size(), columns=['count'])
movies_count_df.head()

# plot rating frequency of all movies
# ax = movies_count_df \
#     .sort_values('count', ascending=False) \
#     .reset_index(drop=True) \
#     .plot(
#         figsize=(12, 8),
#         title='Rating Frequency of All Movies',
#         fontsize=12
#     )
# ax.set_xlabel("movie Id")
# ax.set_ylabel("number of ratings")