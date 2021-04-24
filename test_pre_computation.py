from list_creator import get_database_clean
from pre_computation import save_top_n_movies_by_popularity

df_big = get_database_clean()
save_top_n_movies_by_popularity(similarities_df=df_big, n=10, path_to_save="../Data/similarity_little.csv")
