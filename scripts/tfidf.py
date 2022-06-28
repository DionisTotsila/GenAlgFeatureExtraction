import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pprint import pprint
# remove <num> from data and create "articles" (instead of words they contain str numbers that will be used for the tfidf matrix)
def create_article(row):
    # get data cell
    text  = row['data']
    #remove <num> substrings from text
    text = re.sub('<[0-9]+>',"",text)

    # fix double spaces occuring
    text = ' '.join(text.split())
    return text

# read train data (only the "articles")
df = pd.read_csv("data/train_df.csv", usecols=["data"])

# create row containing only the word ids per article
df["clean"] = df.apply(create_article, axis=1)
pprint(df["clean"][114])
#init tfidf vectorizer
Transformer = TfidfVectorizer(vocabulary =[str(x) for x in range(8520)] )

#gereate tfidf dense matrix matrix
tfidf = Transformer.fit_transform(df.clean.values.astype('U')).todense()


# generate tfidf average vector and store to npz array
mean_vec = tfidf.mean(0)
print(mean_vec.shape)


# create dictionary containing all words (ids) and their corresponding mean tf-idf value
voc_dic = Transformer.vocabulary_

i = 0
for key in voc_dic:
    voc_dic[key] = mean_vec[0, i]
    i+=1

# store dict to file
file = open("data/tfidf.pkl","wb")
pickle.dump(voc_dic, file)
file.close()
