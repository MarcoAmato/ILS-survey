from src.similarities_util import read_lists_of_int_from_csv, PATH_TO_DATA_FOLDER


def test_read_lists_of_int_from_csv():
    read_lists_of_int_from_csv(PATH_TO_DATA_FOLDER + "similar_movies.csv")


if __name__ == "__main__":
    test_read_lists_of_int_from_csv()
