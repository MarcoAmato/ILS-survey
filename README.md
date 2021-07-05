Abstract
=======
Given a dataset of movies information and another dataset of similarity measurements between each movie, the goal of this project is creating lists of movies in various ways, and measuring the ILS of those.

Inside the folder data/lists_of_movies there are various folders, each one corresponding to a series of lists retrieved in a particular manner.
Inside each folder in data/lists_of_movies there are the following .csv files:
- **lists.csv** series of lists of movies, where each list is a series of ids of movies separated by commas. The different lists separate between each other by a new line.
- **ids.csv** ids of all the movies contained in the various lists in *lists.csv*
- **similarities.csv** dataframe which contains [Plot:LDA,Plot:cos,Genre:Jacc,similarity] pairwise similarities for all the for the movies in ids.csv. 
N.B The column "similarity" represents the mean of all similarity measurements for the two movies
- **dataframe_lists.csv** Dataframe where each row contains a list of movies and the corresponding ILS via [Genre:Jacc, Plot:LDA, Plot:cos, mean similarity] for the movies. There is also a label for every movie, this will be displayed in the plot.


N.B.
- We will refer to a dataframe called similarity_mpg.
What MPG stands for is Mean (similarity), Plot:LDA (similarity), Genre:JACC (similarity), which are the 3 different similarity measures for this dataframe. 
- We will refer to another dataframe called similarity_mp2g.
What MP2G stands for is Mean (similarity), Plot:LDA (similarity), Plot:cos (similarity), Genre:JACC (similarity.), which are the 4 different similarity measures for this dataframe. 

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
This file writes files in the data directory, those files will be used to generate the lists with ILS.

What it does:
- The data in the first dataset **all_similarities.csv** is processed, and a file called **similarity_mpg.csv** is created and put into data/top100/. This csv contains a dataframe of columns ['movie1', 'movie2', 'similarity', 'Plot:LDA', 'Genre:Jacc'], where movie1 is the id of the first movie, movie2 is the id of the second movie, similarity is the mean of the similarity measurements, and 'Plot:LDA' and 'Genre:Jacc' are additional similarity measurements. 
- A file called all_movies_ids.csv is created and put into data/. It contains the list of ids of all the movies in the dataset.
- A file called top_100_movies_ids.csv is created and put into data/top100. It contains the ids of the top 100 movies by popularity.
- A file called similarities_mpg.csv is created and put into data/top100. It contains the similarities for the top 100 movies by popularity.
- A folder called movies containing the JSON files of the top 100 movies by popularity is created and put into data/top100

Setup completed
------

Generate ILS plots for lists
======
Once set up the correct dataset, you can run the file **__main__.py**

You are asked to select a list method which plot will be created.
The plot will follow this standard:
- The x-axis shows a list of labels for the various lists in the selected method (when explicit labels are missing, a series of numbers from 0 to n will be shown)
- For each x value (namely, for each list) there is a corresponding dot that has y value equals to the ILS of the list computed by mean similarity.
- It is shown a blue vertical line that has y equals to the mean of 20 random sampled movies in the top 100 movies by popularity. This serves as a reference point.
