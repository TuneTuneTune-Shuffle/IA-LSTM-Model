import pandas as pd
import ast

trackLabelsDir = '/home/assessment1/Downloads/fma_metadata/tracks.csv'
trackgenresDir = '/home/assessment1/Downloads/fma_metadata/genres.csv'

# Header = 0,1 because the data is stored with a column title and a sub-column title
# skiprows = 2 (row = 3) because this row has no value, used only to spefify the column name "track_id", "track_genre_top", "track_genres"
df_tracks = pd.read_csv(trackLabelsDir, header=[0,1], skiprows=[2])
df_genres = pd.read_csv(trackgenresDir)

# Creamos carpetas por genero si no existe y si existe agregamos la cancion a esa carpeta

df_tracks.columns = ['track_id' if i == 0 else f'{col[0]}_{col[1]}' for i, col in enumerate(df_tracks.columns)]
trackID_Label = df_tracks[['track_id', 'track_genre_top', 'track_genres']]


def get_top_level_genre_title(genre_id, df_genres):
    visited = set()

    while True:
        if genre_id in visited:
            # Prevent infinite loops in malformed data
            return "ignore"
        visited.add(genre_id)

        row = df_genres[df_genres['genre_id'] == genre_id]
        if row.empty:
            return "ignore"
        
        current_id = row['genre_id'].iloc[0]
        top_level = row['top_level'].iloc[0]

        if current_id == top_level:
            return row['title'].iloc[0]
        
        genre_id = top_level  # Move one level up


def isGenre(genre_row):
    if pd.notna(genre_row['track_genre_top']):
        return genre_row['track_genre_top']
    
    genre_str = genre_row['track_genres']
    if pd.isna(genre_str):
        return "ignore"

    mainGenre = ast.literal_eval(genre_str)
    if not mainGenre:
        return "ignore"

    genre_id = int(mainGenre[0])
    return get_top_level_genre_title(genre_id, df_genres)


def getGenre():
    trackID_Label['track_genre_top'] = trackID_Label.apply(isGenre,  axis=1)

    return trackID_Label[['track_id', 'track_genre_top']]

#print(trackID_Label.head(50))

#print(trackID_Label[trackID_Label['track_id'] == 613])
#print(trackID_Label)
#print(len(trackID_Label))

    

