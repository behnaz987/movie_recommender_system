import traceback

import numpy as np
import pandas as pd
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.neighbors import NearestNeighbors


class ItemBased(APIView):
    def get(self, request):
        movieName = request.data['movie_name']
        try:
            rates = pd.read_csv('./movie_recommender_system/files/ratings.csv')
            rates.drop(['timestamp'], axis=1, inplace=True)
            movies = pd.read_csv('./movie_recommender_system/files/movies.csv')

            def item_based(movie_name):
                ratings_df = pd.pivot_table(rates, index='userId', columns='movieId', aggfunc=np.max)
                ratings_df.head()
                ratedmovies = pd.merge(rates, movies, on='movieId')
                df_movie_users_series = int(list(movies.loc[movies['title'] == movie_name]['movieId'])[0])

                def get_other_movies(movie_name):
                    # get all users who watched a specific movie
                    df_movie_users = pd.DataFrame(df_movie_users_series, columns=['userId'])
                    other_movies = pd.merge(df_movie_users, ratedmovies, on='userId')
                    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values(
                        'userId',
                        ascending=False)
                    other_users_watched['perc_who_watched'] = round(
                        other_users_watched['userId'] * 100 / other_users_watched['userId'][0], 1)
                    return other_users_watched[:10]

                get_other_movies(movieName)
                # only include movies with more than 10 ratings
                avg_movie_rating = pd.DataFrame(rates.groupby('movieId')['rating'].agg(['mean', 'count']))
                avg_movie_rating['movieId'] = avg_movie_rating.index
                movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count'] >= 10]
                print('Number of movies which has more than 10 rates: ', len(movie_plus_10_ratings))
                movie_plus_10_ratings.index.name = None
                filtered_ratings = pd.merge(movie_plus_10_ratings, rates, on="movieId")
                print('the number of records in new rates table: ', len(filtered_ratings))
                filtered_ratings.head()
                movie_wide = filtered_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
                movie_wide.head()
                model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
                model_knn.fit(movie_wide)

                # Gets the top 10 nearest neighbours got the movie
                def get_similar_movies(movieid):
                    query_index_movie_ratings = movie_wide.loc[movieid, :].values.reshape(1, -1)
                    distances, indices = model_knn.kneighbors(query_index_movie_ratings, n_neighbors=11)
                    final = {}
                    for i in range(0, len(distances.flatten())):
                        get_movie = (movies.loc[movies['movieId'] == movieid]['title'])
                        if i == 0:
                            print('Recommendations for {0}\n'.format(get_movie))
                        else:
                            indices_flat = indices.flatten()[i]
                            get_movie = (
                                list(movies.loc[movies['movieId'] == movie_wide.iloc[indices_flat, :].name]['title']))
                            final[i] = get_movie[0]
                    print(final)
                    return final

                return get_similar_movies(df_movie_users_series)

            return Response(item_based(movie_name=movieName), status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return Response(status=status.HTTP_404_NOT_FOUND)


class GetMovie(APIView):
    def get(self, request):
        try:
            info = {}
            movies = pd.read_csv('./movie_recommender_system/files/movies.csv')
            movieId = request.data['movie_id']
            info["movieTitle"] = (movies.iloc[int(movieId) - 1, 1])
            info["movieGenres"] = (movies.iloc[int(movieId) - 1, 2])
            return Response(info, status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return Response(status=status.HTTP_404_NOT_FOUND)
