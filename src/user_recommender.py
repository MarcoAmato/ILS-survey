import math
import random
from typing import List, Optional, Set

import pandas as pd
from pandas import DataFrame

# ratings.csv [userId, movieId, rating, timestamp]
# movies.csv [movieId, title, genres]
from src.similarities_util import PATH_TO_DATA_FOLDER, read_lists_of_int_from_csv, PATH_TO_TOP_100_MOVIES_ID, \
    read_movie_ids_from_csv

NUM_USERS = 610
NUM_NEIGHBOURS = 10
USERS_TO_SAMPLE = 10

PATH_TO_MOVIELENS = PATH_TO_DATA_FOLDER + "movieLens/"
PATH_TO_RATINGS_RAW = PATH_TO_MOVIELENS + "ratings_small.csv"
PATH_TO_RATINGS_NEW = PATH_TO_MOVIELENS + "ratings_new.csv"


# returns the input if it is a valid id, none otherwise
def get_id_from_input():
    user_input = input("Insert a user id to get the recommendations\n")
    try:
        val = int(user_input)  # tries to convert input to int
        if val < 1 or val > NUM_USERS:  # it is not an id in the correct range
            print(f"Id must be between 1 and {NUM_USERS}")
            return None
        return val  # okay
    except ValueError:  # input is not an integer
        print(f"Expected an integer id, got '{user_input}'")
        return None


# returns the movies that "user_id" has rated
def get_movies_rated_by_user(user_id: int, df_movies: DataFrame, df_ratings: DataFrame) -> DataFrame:
    df_movies_rated_by_user = df_movies.merge(df_ratings)  # we join on the columns "movie_id"
    df_movies_rated_by_user = df_movies_rated_by_user[df_movies_rated_by_user['userId'] == user_id]  # we select
    # the columns with the input userId as "userId"
    return df_movies_rated_by_user


def get_movies_and_ratings(movies, ratings):
    movies_and_ratings_df = movies.merge(ratings)  # join on common column "movie_id"
    return movies_and_ratings_df


def create_user_movie_pivot_table(movie_ratings: DataFrame) -> DataFrame:
    # create pivot table with userId, movieId, ratings
    table = pd.pivot_table(movie_ratings, index=['userId', "movieId"], values=['rating'])
    # (table.loc[1, 367]) to get element rating of user 1 and movie 367
    return table


def get_similarity(user1_ratings: DataFrame, user2_ratings: DataFrame) -> Optional[float]:
    user1_ratings.rename(columns={"rating": "user1"}, inplace=True)  # rename ratings column to 'user1'
    user2_ratings.rename(columns={"rating": "user2"}, inplace=True)  # rename ratings column to 'user2'
    df: DataFrame = pd.concat([user1_ratings, user2_ratings], axis=1)

    common_rated_movies = 0  # number of common ratings

    for index, row in df.iterrows():
        # print(type(row))
        # print(row)
        # print(row['user1']['rating'])
        # exit()
        if row['user1'] >= 0 and row['user2'] >= 0:  # They both rated the movie
            common_rated_movies += 1
        if common_rated_movies >= 3:
            break

    if common_rated_movies >= 3:
        # gets the correlation dataframe of the users similarities
        pearson_corr: float = df.corr(method="pearson", min_periods=1).loc["user1", "user2"]

        return pearson_corr
    else:
        return None  # not enough data to have a similarity


def get_user_recommendations(user_id: int) -> List[int]:
    """
    Returns recommended movies for user_id using nearest k-neighbours technique
    :param user_id: id of user to be recommended
    :return: list of recommended movies_ids
    """
    df_movies: DataFrame = pd.read_csv("../dataset/ml-latest-small/movies.csv")
    df_ratings: DataFrame = pd.read_csv("../dataset/ml-latest-small/ratings.csv")
    df_ratings.drop("timestamp", axis=1)  # remove column "timestamp"

    movies_rated_by_user: DataFrame = get_movies_rated_by_user(user_id, df_movies, df_ratings)

    movies_and_ratings: DataFrame = df_movies.merge(df_ratings)  # all ratings with related movie information

    print("Movie rated by user")
    print(movies_rated_by_user.iloc[:15][['title', "genres"]])  # we print the title and genre of the first 15 movies

    # list of ids of movies rated by userId
    movie_ids: List[int] = movies_rated_by_user[(movies_rated_by_user.userId == user_id)]['movieId'].to_list()

    # we create the ratings table
    ratings_table: DataFrame = create_user_movie_pivot_table(movies_and_ratings)

    user_similarity_list = []
    for curr_user_id in range(1, NUM_USERS):
        if user_id == curr_user_id:
            continue  # the rating with itself is ignored
        # similarity between input_user_id and user_id

        similarity = get_similarity(ratings_table.loc[user_id], ratings_table.loc[curr_user_id])

        if similarity is not None and not math.isnan(similarity):
            # there are at least than 3 ratings in the same movies
            user_similarity_list.append({"userId": user_id, "similarity": similarity})

    #  list sorted by descending similarity
    sorted_similarities = sorted(user_similarity_list, key=lambda k: k["similarity"], reverse=True)

    neighbours_ratings: DataFrame = pd.DataFrame(columns=["movieId", "rating"])

    for nth_neighbour in range(NUM_NEIGHBOURS):
        neighbour_id = sorted_similarities[nth_neighbour]["userId"]
        # add all the ratings for the neighbour_id
        neighbours_ratings = \
            pd.concat([neighbours_ratings, movies_and_ratings[(movies_and_ratings.userId == neighbour_id)]])

    neighbours_ratings_grouped_by_movie = neighbours_ratings[['movieId', 'rating']].copy().groupby(by='movieId')

    recommended_movies = []
    for movie_id, movie_group in neighbours_ratings_grouped_by_movie:
        movie_with_rating = {"movieId": movie_id, "rating": movie_group['rating'].mean()}
        recommended_movies.append(movie_with_rating)

    # sort movies by descending rating
    recommended_movies = sorted(recommended_movies, key=lambda k: k['rating'], reverse=True)

    recommended_movies_ids: List[int] = []
    for record in recommended_movies[:10]:
        recommended_movies_ids.append(record['movieId'])

    return recommended_movies_ids


def get_users_ids() -> List[int]:
    # users are random for now, could be selected differently
    users = set(ratings.userId.tolist())  # set of user ids
    return random.sample(users, USERS_TO_SAMPLE)


def pre_compute_ratings(path_to_movies_ids: str):
    """
    Starting from the movies dataframe returns a smaller one containing the movies in path_to_movies_ids
    @param path_to_movies_ids: path to file containing movie ids
    @type path_to_movies_ids: str
    """
    ids = read_movie_ids_from_csv(path_to_movies_ids)
    ratings: DataFrame = pd.read_csv(PATH_TO_RATINGS_RAW)
    ratings_new = ratings[ratings.movieId.isin(ids)]

    print(ratings_new)
    print(len(ratings_new))

    ratings_new.to_csv(PATH_TO_RATINGS_NEW)


if __name__ == '__main__':
    pre_compute_ratings(PATH_TO_TOP_100_MOVIES_ID)

    ratings = pd.read_csv(PATH_TO_RATINGS_NEW)
    print(ratings)
    exit()


    users_sampled = get_users_ids()
    recommendations: List[List[int]] = []
    for user in users_sampled:
        recommendations.append(get_user_recommendations(user))

    print(recommendations)
