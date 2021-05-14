from typing import List, Dict, Tuple

from pandas import DataFrame

import main


def get_similarities_containing_values(similarities_df: DataFrame, values_to_look_for: List[int]) -> \
        Dict[Tuple[int, int], List[str]]:
    """
    Returns a dict, key is movie1 and movie2, value are list of columns that contain values
    :param similarities_df: dataframe of similarities
    :param values_to_look_for: list of values to look for in similarity columns
    """
    dict_similarities_with_values: Dict[Tuple[int, int], List[str]] = {}

    for index, row in similarities_df.iterrows():
        list_of_column_with_values: List[str] = []
        for sim_column in main.COLUMNS_SIMILARITY:  # iterate similarity columns
            if row[sim_column] in values_to_look_for:  # row contains value
                # add column to list_of_column_with_values
                list_of_column_with_values.append(sim_column + " = " + str(int(row[sim_column])))
        if len(list_of_column_with_values) > 0:  # there are columns containing 'values' in the row
            movieId1: int = row["validation$r1"].split(".")[0]
            movieId2: int = row["validation$r2"].split(".")[0]
            dict_similarities_with_values[movieId1, movieId2] = list_of_column_with_values

    return dict_similarities_with_values


if __name__ == "__main__":
    similarities: DataFrame = main.get_dataframe_movie_ids_and_similarities(500)
    similarities_value_suspect: Dict[Tuple[int, int], List[str]] = \
        get_similarities_containing_values(similarities, [1, 0, -1])
    # pd.set_option('display.max_columns', None)
    print(similarities)

    for key in similarities_value_suspect:
        movie1: str = f"Movie {key[0]} name: \n\t" + \
                      main.get_name_of_movie(main.get_movie(key[0]))
        print(movie1)
        movie2: str = f"Movie {key[1]} name: \n\t" + \
                      main.get_name_of_movie(main.get_movie(key[1]))
        print(movie2)
        print("Columns with suspicious values")
        print(similarities_value_suspect[key])
        print("--------------------")
