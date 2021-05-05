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
This file writes files in the data directory, those files will be used to get the lists with ILD.

What it does:
- The data in the first dataset **all_similarities.csv** is processed, and a file called **similarity_mpg.csv** is created and put into data/top100/. This csv contains a dataframe of columns ['movie1', 'movie2', 'similarity', 'Plot:LDA', 'Genre:Jacc'], where movie1 is the id of the first movie, movie2 is the id of the second movie, similarity is the mean of the similarity measurements, and 'Plot:LDA' and 'Genre:Jacc' are additional similarity measurements. 
- A file called all_movies_ids.csv is created and put into data/. It contains the list of ids of all the movies in the dataset.
- A file called top_100_movies_ids.csv is created and put into data/top100. It contains the ids of the top 100 movies by popularity.
- A file called similarities_mpg.csv is created and put into data/top100. It contains the similarities for the top 100 moviesby popularity.
- A folder called movies containing the JSON files of the top 100 movies by popularity is created and put into data/top100

Setup completed
------

Generate lists with ILS
======
Once set up the correct dataset you can run the file **main.py**

- A list of 10 movies, called**sample_list_of_movies**, is generated taking random elements from the top 100 movies by popularity.
- the ILS of **sample_list_of_movies** is computed and printed in the following ways:
    - mean of similarity measurements
    - Plot:LDA
    - Genre:JACC
    - mean of Plot:LDA and Genre:JACC
