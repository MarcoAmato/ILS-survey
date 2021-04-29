Before starting
=======
It is necessary to prepare a folder for the datasets. In order to run the files the folder should be organised as follows:
1)Inside the project root it should be created a folder called 'Data'.
2)Inside the Data folder it should be the first version of the dataset of similarity measurements, and it should be called **pred2-incl-all_all.csv**
3)Inside the Data folder it should be folder containing the JSON files of movies.
    -The folder should be called **extracted_content_ml-latest**
    -The movies JSONs should be called **<movieId>.json**
    
Files purpose
=======

pre_computation.py
--------
This file writes optimized files in the Data directory.
list_creator.py assumes that these files are already created.

The main calls:
-**write_light_dataframe(PATH_TO_NEW_SIMILARITY)**: The data in the first dataset **pred2-incl-all_all.csv** is processed, and a file called clean_similarity.csv is created in the Data folder. This csv contains a dataframe of columns ['movie1', 'movie2', 'similarity'], where movie1 is the id of the first movie, movie2 is the id of the second movie and similarity is the mean of the similarity measurements for the two movies. 
-write_all_movies_ids(PATH_TO_ALL_MOVIES_ID): A file called all_movies_ids.csv is created in the Data folder. It contains the list of ids of all the movies in the dataset.
-write_top_n_movies_by_popularity(10, PATH_TO_TOP_10_MOVIES_ID): A file called top_10_movies_ids.csv is created in the Data folder. It contains the ids of the top 10 movies by similarity.

list_creator.py
--------
**NB, before running this file it should first be run pre_computation.py in order to have updated datasets**

This file allows getting lists of movies with the related ILS.

The main calls **test_top_10_movies()**.
From the top 10 movies by popularity, it samples a list of length: MOVIES_LIST_LENGTH and prints it alongside the ILS