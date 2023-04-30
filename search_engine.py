import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def search(query) :

    # Load competitions data
    competitions = pd.read_csv('competitions.csv')
    # Create vectorizer object and fit to titles
    vectorizer = TfidfVectorizer()
    vec_fitted = vectorizer.fit_transform(competitions['title'])

    # Create a vector from the query
    query_vec = vectorizer.transform([query.lower()])
    # Sort the competitions by similarity score
    res = competitions.assign(similarity = cosine_similarity(query_vec, vec_fitted).flatten()).sort_values('similarity', ascending = False).reset_index(drop = True)

    # If max similarity is low, assume no results exist
    if res['similarity'][0] < 0.01 :
        print('No results ...')
        output, flag = None, False

    # If max similarity is near perfect, assign the competition
    elif res['similarity'][0] > 0.99 :
        print(res['Competition'][0], res['CODE'][0])
        output, flag = res.head(1), True

    # If max similarity is between the two, ask for confirmation to the user
    else :
        resp = input(f"{res['Competition'][0]} ({res['Country'][0]}) ?  ([y], n)")
        if resp == '' or resp == 'y' or resp == 'Y' :
            output, flag = res.head(1), True
        else :
            output, flag = None, False
            print('Aborted ...')

    # Output is 0 if no results or the competition info row
    return (output, flag)

