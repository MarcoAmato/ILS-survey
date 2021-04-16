import pandas as pd

PATH_TO_DATA_FOLDER = "../Data/"
PATH_TO_RATINGS = PATH_TO_DATA_FOLDER + "pred2-incl-all_all.csv"
PATH_TO_JSON = PATH_TO_DATA_FOLDER + "extracted_content_ml-latest/"
COLUMNS_SIMILARITY = ['Title:LEV', 'Title:JW', 'Title:LCS', 'Title:BI',
                      'Title:LDA', 'Image:EMB', 'Image:BR', 'Image:SH', 'Image:CO',
                      'Image:COL', 'Image:EN', 'Plot:LDA', 'Plot:cos', 'Genre:LDA',
                      'Genre:Jacc', 'Stars:Jacc', 'Directors:Jacc', 'Date:MD', 'Tag:Genome',
                      'SVD']


def get_database_clean(num_rows: int):
    #  returns a pandas dataframe containing the columns [validation$r1, validation$r2] hence, the film ids, and the
    #  similarity measurements of the two
    interested_columns = ["validation$r1", "validation$r2"] + COLUMNS_SIMILARITY
    return pd.read_csv(PATH_TO_RATINGS, nrows=num_rows, sep="\t", usecols=interested_columns)


def get_columns_of_dataframe(dataframe: pd.DataFrame):
    return dataframe.columns


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
    film_paths: list[str, str] = get_film_paths(similarity_row)
    print("First movie: \n\t"+get_name_of_film(film_paths[0]))
    print("Second movie: \n\t"+get_name_of_film(film_paths[1]))

    similarity_values = similarity_row[COLUMNS_SIMILARITY]
    print("Similarity value: \n\t" + str(sum(similarity_values)/len(similarity_values)))
    exit()


if __name__ == '__main__':
    df = get_database_clean(100)
    mean_similarity: float = get_mean_similarity(df.iloc[1])
    # print(get_columns_of_dataframe(df))
    # test_validation1 = df.iloc[1]["validation$r1"]
    # print(test_validation1)
    # print(get_film(test_validation1))
