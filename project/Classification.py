import os
import string
import re
import pymorphy3
import pandas as pd

from joblib import dump
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

morph = pymorphy3.MorphAnalyzer()

cur_dir = os.getcwd()
test_dat = os.path.join(cur_dir, 'train_dataset.csv')
test_dat = pd.read_csv(test_dat)

garbage = string.punctuation + string.digits + 'club' + 'id' + '—'

# Deleting TRASH and NON-ESSENTIAL INFO
def prepare(txt):
    prep_text = ''
    txt = txt.translate(str.maketrans('', '', garbage))
    for word in txt.split():
        new_line = ''
        if word not in stopwords.words("russian"):
            word = morph.parse(word)[0].normal_form
            new_line += word + ' '
        else:
            new_line += ''
        prep_text += '' + new_line.lower()
    re.sub(r'\s+', ' ', prep_text, flags=re.I)
    return prep_text


# Teaching MODEL
def teach(file):
    file['preproccessed'] = list(map(prepare, file['Текст инцидента']))

    x = file['preproccessed']
    y = file['Тема']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    file['Тема'].unique()

    logreg = Pipeline([('vec', CountVectorizer()),
                       ('tfdf', TfidfTransformer()),
                       ('lr', LogisticRegression(n_jobs=1, C=1e5)),
                       ])

    logreg.fit(x_train, y_train)

    model_file = 'trained_model.joblib'
    dump(logreg, model_file)


'''match input('Необходимо провести новое обучение модели?(Y/N):\n'):
    case 'N', 'n':
        print('Ok')
    case 'Y', 'y':
        teach(test_dat)'''
