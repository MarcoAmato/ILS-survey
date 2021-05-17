import random
from typing import List

from pandas import DataFrame

from src.similarities_util import read_movie_ids_from_csv, PATH_TO_TOP_100_MOVIES_ID, \
    get_movie_dataframe_from_id, get_similar_movies, get_similarity_dataframe, PATH_TO_SIM_100_SIMILARITIES, \
    print_ILS_measures


def print_similar_movies_ILS() -> None:
    id_movies_top_100: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    movie_id: int = random.sample(id_movies_top_100, 1)[0]
    movie_df: DataFrame = get_movie_dataframe_from_id(movie_id)
    movies_plus_similar: List[int] = get_similar_movies([movie_df])
    similarity_df: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_SIMILARITIES)
    print_ILS_measures(movies_plus_similar, similarity_df)


if __name__ == "__main__":
    print_similar_movies_ILS()
