from typing import List

from pandas import DataFrame

from src.similarities_util import read_lists_of_int_from_csv, PATH_TO_DATA_FOLDER, get_movies_df_from_json_folder, \
    PATH_TO_TOP_100_JSON, get_movies_with_name, PATH_TO_JSON, get_movies_df_from_json_folder_where_name_in_title, \
    convert_tbdb_to_movieId, get_similarities_with_condition, get_similarity_dataframe, PATH_TO_SIM_100_MP2G, \
    does_row_contain_movies, print_lists_in_file_ILS


def test_read_lists_of_int_from_csv():
    read_lists_of_int_from_csv(PATH_TO_DATA_FOLDER + "similar_movies.csv")


def test_get_movies_df_from_json_folder():
    list_of_movies: List[DataFrame] = get_movies_df_from_json_folder(PATH_TO_TOP_100_JSON)
    print(list_of_movies)


def test_get_movies_with_name():
    movies_toy_story: List[DataFrame] = get_movies_with_name("Toy Story", PATH_TO_JSON)
    for movie in movies_toy_story:
        print(movie["tmdb"]["title"])


def test_get_movies_df_from_json_folder_where_name_in_title(name_movie: str):
    print(f"Movies containing {name_movie}")
    movies_toy_story: List[DataFrame] = get_movies_df_from_json_folder_where_name_in_title(path_to_json=PATH_TO_JSON,
                                                                                           name=name_movie)
    movies_titles: List[str] = []
    movies_ids: List[int] = []

    for movie in movies_toy_story:
        movies_titles.append(movie["tmdb"]["title"])
        movies_ids.append(movie["tmdb"]["id"])

    movie_ids_converted: List[int] = convert_tbdb_to_movieId(movies_ids)

    print(movie_ids_converted)

    for i in range(0, len(movies_titles)):
        print(f"{movies_titles[i]}, {convert_tbdb_to_movieId([movies_ids[i]])}")
    print("----------")


def test_get_similarities_with_condition():
    print("test_get_similarities_with_condition")
    similarities_top_100: DataFrame = get_similarity_dataframe(PATH_TO_SIM_100_MP2G)
    list_of_movies: List[int] = [1, 5349, 52722]
    similarities_new: DataFrame = get_similarities_with_condition(similarities_top_100, list_of_movies,
                                                                  does_row_contain_movies)
    print(similarities_new)


if __name__ == "__main__":
    print_lists_in_file_ILS()
