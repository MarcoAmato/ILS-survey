import pandas
import pandas as pd

PATH_TO_DATA_FOLDER = "../Data/"
PATH_TO_RATINGS = PATH_TO_DATA_FOLDER + "pred2-incl-all_all.csv"
PATH_TO_JSON = PATH_TO_DATA_FOLDER + "extracted_content_ml-latest/"
COLUMNS_SIMILARITY = {'Title:LEV', 'Title:JW', 'Title:LCS', 'Title:BI',
                      'Title:LDA', 'Image:EMB', 'Image:BR', 'Image:SH', 'Image:CO',
                      'Image:COL', 'Image:EN', 'Plot:LDA', 'Plot:cos', 'Genre:LDA',
                      'Genre:Jacc', 'Stars:Jacc', 'Directors:Jacc', 'Date:MD', 'Tag:Genome',
                      'SVD'}


def get_database_clean(num_rows: int):
    #  returns a pandas dataframe containing the columns [validation$r1, validation$r2] hence, the film ids, and the
    #  similarity measurements of the two
    interested_columns = {"validation$r1", "validation$r2"}.union(COLUMNS_SIMILARITY)
    return pd.read_csv(PATH_TO_RATINGS, nrows=num_rows, sep="\t", usecols=interested_columns)


def get_columns_of_dataframe(dataframe: pd.DataFrame):
    return dataframe.columns


def get_similarity_rows_of_movie(dataframe: pd.DataFrame, movie_id: str):
    sim_df_of_movie = pandas.DataFrame(columns=dataframe.columns)
    for i, sim_row in dataframe.iterrows():
        if sim_row["validation$r1"] == movie_id or sim_row["validation$r2"] == movie_id:
            sim_df_of_movie = sim_df_of_movie.append(sim_row, ignore_index=True)
    return sim_df_of_movie


def get_film_paths(similarity_row: pd.Series):
    film1 = (similarity_row["validation$r1"])
    film2 = (similarity_row["validation$r2"])
    return [film1, film2]


def get_film(path: str):
    return pd.read_json(PATH_TO_JSON + path)


def get_name_of_film(film: str):
    film = get_film(film)
    return film["tmdb"]["title"]


def get_mean_similarity(similarity_row: pd.Series):
    similarity_values = similarity_row[COLUMNS_SIMILARITY]
    return sum(similarity_values) / len(similarity_values)


if __name__ == '__main__':
    similarity_dataframe = get_database_clean(3)
    mean_similarities = list()
    for index in range(len(similarity_dataframe.index)):
        mean_similarities.append(get_mean_similarity(similarity_dataframe.iloc[index]))
    similarity_dataframe['mean_similarity'] = mean_similarities

    movie_id = (similarity_dataframe.iloc[1]["validation$r2"])
    print(get_similarity_rows_of_movie(similarity_dataframe, movie_id))

    # print(get_columns_of_dataframe(df))
    # test_validation1 = df.iloc[1]["validation$r1"]
    # print(test_validation1)
    # print(get_film(test_validation1))
