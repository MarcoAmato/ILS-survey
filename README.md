Abstract
=======
Given a dataset of movies information and another dataset of similarity measurements between each movie, the goal of this project is creating lists from the 100 most popular  movies with relative ILD measurements.

Project setup
=======
Fill data folder
------
The folder 'data' should be filled with the following:
- **movies_jsons** A folder containing a set of jsons with movies information. Each file should be called **\<movieId\>.json**
- **all_similarities.csv** A csv containing a dataset with similarity measurements for each pair of movie.
    The similarity measures are: *Title:LV, Title:JW, Title:LCS, Title:BI, Title:LDA, Image:EMB, Image:BR, Image:SH, Image:CO, Image:COL, Image:EN, Plot:TFIDF, Plot:LDA, Genre:LDA, Dir:Jacc, Date:MD, Stars:JACC, Tag, SVD*

Run pre_computation.py
------
This file writes optimized files in the data directory.

The main calls:
- **write_light_dataframe(PATH_TO_NEW_SIMILARITY)**: The data in the first dataset **pred2-incl-all_all.csv** is processed, and a file called clean_similarity.csv is created in the Data folder. This csv contains a dataframe of columns ['movie1', 'movie2', 'similarity'], where movie1 is the id of the first movie, movie2 is the id of the second movie and similarity is the mean of the similarity measurements for the two movies. 
- write_all_movies_ids(PATH_TO_ALL_MOVIES_ID): A file called all_movies_ids.csv is created in the Data folder. It contains the list of ids of all the movies in the dataset.
- write_top_n_movies_by_popularity(10, PATH_TO_TOP_10_MOVIES_ID): A file called top_10_movies_ids.csv is created in the Data folder. It contains the ids of the top 10 movies by similarity.


You can start
======
Once set up the correct dataset you can run the file **list_creator.py**

This file allows getting lists of movies with the related ILS.

The main calls **test_top_10_movies()**.
From the top 10 movies by popularity, it samples a list of length: MOVIES_LIST_LENGTH and prints it alongside the ILS
