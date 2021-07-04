import math
import random
from typing import List, Optional, Set

import pandas as pd
from pandas import DataFrame

# ratings.csv [userId, movieId, rating, timestamp]
# movies.csv [movieId, title, genres]
from src.similarities_util import PATH_TO_DATA_FOLDER, read_movie_ids_from_csv

NUM_USERS = 610
NUM_NEIGHBOURS = 10
USERS_TO_SAMPLE = 10
MOVIES_PER_LIST = 7

PATH_TO_MOVIELENS = PATH_TO_DATA_FOLDER + "movieLens/"
PATH_TO_RATINGS_RAW = PATH_TO_MOVIELENS + "ratings_small.csv"
PATH_TO_MOVIES_RAW = PATH_TO_MOVIELENS + "movies.csv"

PATH_TO_RATINGS_NEW = PATH_TO_MOVIELENS + "ratings_new.csv"
PATH_TO_MOVIES_NEW = PATH_TO_MOVIELENS + "movies_new.csv"

# dataframes shared by the functions
df_movies: DataFrame = pd.read_csv(PATH_TO_MOVIES_NEW)
df_ratings: DataFrame = pd.read_csv(PATH_TO_RATINGS_NEW)
movies_and_ratings: DataFrame = df_movies.merge(df_ratings, on="movieId")  # all ratings with related movie information
ratings_table: DataFrame = pd.pivot_table(movies_and_ratings, index=['userId', "movieId"], values=['rating'])
users: Set[int] = set(df_ratings.userId.tolist())  # set of user ids


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
def get_movies_rated_by_user(user_id: int) -> DataFrame:
    df_movies_rated_by_user = df_movies.merge(df_ratings, on="movieId")  # we join on the columns "movie_id"
    df_movies_rated_by_user = df_movies_rated_by_user[df_movies_rated_by_user['userId'] == user_id]  # we select
    # the columns with the input userId as "userId"
    return df_movies_rated_by_user


def get_movies_and_ratings(movies, ratings):
    movies_and_ratings_df = movies.merge(ratings, on="movieId")  # join on common column "movie_id"
    return movies_and_ratings_df


def get_similarity(user1_ratings: DataFrame, user2_ratings: DataFrame) -> Optional[float]:
    user1_ratings.rename(columns={"rating": "user1"}, inplace=True)  # rename ratings column to 'user1'
    user2_ratings.rename(columns={"rating": "user2"}, inplace=True)  # rename ratings column to 'user2'
    df: DataFrame = pd.concat([user1_ratings, user2_ratings], axis=1)

    common_rated_movies = 0  # number of common ratings

    for index, row in df.iterrows():
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


def get_user_recommendations(user_id: int) -> Optional[List[int]]:
    """
    Returns recommended movies for user_id using nearest k-neighbours technique, or None when data not available
    :param user_id: id of user to be recommended
    :return: list of recommended movies_ids
    """
    # ratings_user: DataFrame = get_movies_rated_by_user(user_id)

    sorted_similarities = get_similarities_for_user(user_id)

    if len(sorted_similarities) < NUM_NEIGHBOURS:
        return None  # not enough similarities to have NUM_NEIGHBOURS

    neighbours_ratings = get_nearest_neighbours(sorted_similarities)

    neighbours_ratings_grouped_by_movie = neighbours_ratings[['movieId', 'rating']].copy().groupby(by='movieId')

    recommended_movies = []
    for movie_id, movie_group in neighbours_ratings_grouped_by_movie:
        movie_with_rating = {"movieId": movie_id, "rating": movie_group['rating'].mean()}
        recommended_movies.append(movie_with_rating)

    # sort movies by descending rating
    recommended_movies = sorted(recommended_movies, key=lambda k: k['rating'], reverse=True)

    recommended_movies_ids: List[int] = []
    for record in recommended_movies[:MOVIES_PER_LIST]:  # add as many movies as MOVIES_PER_LIST
        recommended_movies_ids.append(record['movieId'])

    return recommended_movies_ids


def get_nearest_neighbours(sorted_similarities) -> DataFrame:
    neighbours_ratings: DataFrame = pd.DataFrame(columns=["movieId", "rating"])
    for nth_neighbour in range(NUM_NEIGHBOURS):
        neighbour_id = sorted_similarities[nth_neighbour]["userId"]
        # add all the ratings for the nth_neighbour
        neighbours_ratings = \
            pd.concat([neighbours_ratings, movies_and_ratings[(movies_and_ratings.userId == neighbour_id)]])
    return neighbours_ratings


def get_similarities_for_user(user_id):
    user_similarity_list = []
    for curr_user_id in users:
        if user_id == curr_user_id:
            continue  # the rating with itself is ignored

        similarity = get_similarity(ratings_table.loc[user_id], ratings_table.loc[curr_user_id])

        if similarity is not None and not math.isnan(similarity):
            # there are at least than 3 ratings in the same movies
            user_similarity_list.append({"userId": curr_user_id, "similarity": similarity})
    #  list sorted by descending similarity
    sorted_similarities = sorted(user_similarity_list, key=lambda k: k["similarity"], reverse=True)
    return sorted_similarities


def get_users_ids() -> List[int]:
    # users are random for now, could be selected differently
    return random.sample(users, USERS_TO_SAMPLE)


def pre_compute(path_to_movies_ids: str):
    """
    Starting from the movies dataframe returns a smaller one containing the movies in path_to_movies_ids
    @param path_to_movies_ids: path to file containing movie ids
    @type path_to_movies_ids: str
    """
    ids = read_movie_ids_from_csv(path_to_movies_ids)
    ratings_old: DataFrame = pd.read_csv(PATH_TO_RATINGS_RAW)
    movies_old: DataFrame = pd.read_csv(PATH_TO_MOVIES_RAW)

    movies_new = DataFrame(columns=movies_old.columns)
    ratings_new = DataFrame(columns=ratings_old.columns)

    movies_new = movies_new.append(movies_old[movies_old.movieId.isin(ids)], ignore_index=True)
    ratings_new = ratings_new.append(ratings_old[ratings_old.movieId.isin(ids)], ignore_index=True)
    ratings_new.drop("timestamp", axis=1, inplace=True)  # remove column "timestamp"

    ratings_new.reset_index(drop=True, inplace=True)
    movies_new.reset_index(drop=True, inplace=True)

    ratings_new.to_csv(PATH_TO_RATINGS_NEW)
    movies_new.to_csv(PATH_TO_MOVIES_NEW)


if __name__ == '__main__':
    users_sampled = get_users_ids()
    recommendations: List[List[int]] = []
    for user in users_sampled:
        recommendations.append(get_user_recommendations(user))

    print(recommendations)
