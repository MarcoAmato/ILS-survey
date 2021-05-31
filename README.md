Abstract
=======
Given a dataset of movies information and another dataset of similarity measurements between each movie, the goal of this project is creating lists of movies in various ways, and measuring the ILS of those.

The lists can be created in the following ways:
1) By manually entering the ids of the movies
2) By extracting a random movie from the top 100 and then adding all the similar movies. These similar movies are found in the column "similar" of the json file of the selected movie. 
3) By creating a list of random items from the top 100. The user will provide the number of random items for the list.
4) By computing the ILS for the lists of movies in data/lists_of_movies.csv. This file will contain lists of ids in the following format:
    - Every line contains a list of number characters separated by comma, each one representing an id
    - The end of line signs the end of a list
    
    Example of a valid list_of_movies.csv:


    1,64,5,8
    6,3,98,125
    9,54,65,3,7,5,12,41,98,27,93

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

Generate lists with ILS
======
Once set up the correct dataset, you can run the file **__main__.py**

- You are asked to select a method to create the list. These methods are indicated in the Abstract section.
- The ILS of the list generated is computed and printed in the following ways:
    - mean of similarity measurements
    - Plot:LDA
    - Genre:JACC
    - mean of Plot:LDA and Genre:JACC
