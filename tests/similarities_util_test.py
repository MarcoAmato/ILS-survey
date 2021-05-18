import random
from typing import List

from pandas import DataFrame

from src.similarities_util import read_movie_ids_from_csv, PATH_TO_TOP_100_MOVIES_ID, \
    get_movie_dataframe_from_id, get_similar_movies, get_similarity_dataframe, PATH_TO_SIM_100_SIMILARITIES, \
    print_ILS_measures, PATH_TO_TOP_100_SIMILARITIES_JSON, convert_tbdb_to_movieId, get_movie_name


def print_similar_movies_ILS() -> None:
    id_movies_top_100: List[int] = read_movie_ids_from_csv(PATH_TO_TOP_100_MOVIES_ID)
    while True:
        print("Press enter to sample a movie and look for ILS of similarities")
        input()

        movie_id: int = random.sample(id_movies_top_100, 1)[0]
        movie_df: DataFrame = get_movie_dataframe_from_id(movie_id)
        movie_name: str = get_movie_name(movie_df)
        print(f"sampled movie_id: {movie_id}, name: {movie_name}")

        movies_plus_similar_tmdb: List[int] = get_similar_movies([movie_df])
        movies_plus_similar_movieId: List[int] = convert_tbdb_to_movieId(movies_plus_similar_tmdb)

        similarity_df: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_SIMILARITIES)
        print_ILS_measures(movies_plus_similar_movieId, similarity_df, PATH_TO_TOP_100_SIMILARITIES_JSON)


if __name__ == "__main__":
    print_similar_movies_ILS()
