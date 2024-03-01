'''
Practical Task 2
Let us build a system that will tell you what to watch next based on the word vector similarity of the description of movies.
- Read in the movies.txt file. Each separate line is a description of a different movie.
- Your task is to create a function to return which movies a user would watch next if they have watched Planet Hulk with the description
“Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk in to a shuttle and
launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into
slavery and trained as a gladiator.”
- The function should take in the description as a parameter and return the title of the most similar movie
'''

import spacy

nlp = spacy.load('en_core_web_md')

# Creates dictionary with movies user can watch
movies_dic = {}
file1 = open("movies.txt", "r")
read_content = file1.read()
file1.close()
read_content = read_content.split('\n')
for line in read_content:
    if line:
        line = line.split(':')
        movies_dic[line[0].strip()] = line[1]


def movie_recommendation(desc):
    # This def compares description of last watched movie with descriptions of movies in dictionary
    # Creates list based on similarities in description and returns Title of most similar one
    model_desc = nlp(desc)
    watch_next_list = []
    for movie in movies_dic:
        similarity = nlp(movies_dic[movie]).similarity(model_desc)
        watch_next_list.append([similarity, movie])
    watch_next_list.sort(reverse=True)
    # print(watch_next_list)
    return watch_next_list[0][1]

title_of_watched_movie = 'Planet Hulk'
description_of_watched_movie = """Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth,
    the Illuminati trick Hulk in to a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately,
    Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator."""

print("Bro, if you liked {}, you should deffinitely watch {}.".format(title_of_watched_movie, movie_recommendation(description_of_watched_movie)))
