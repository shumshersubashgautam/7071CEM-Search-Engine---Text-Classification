
# Information Retrieval Subject Text Classifier (STW7071CEM )- Subash Gautam

Task:
Whether as a separate program or integrated with search engine, a subject classification functionality is needed. More specifically, the input is a scientific text and the output is its subject among zero or more of the cases: Health, Sports, Business etc.

This program implements a text classifier using a NLP techniques and a selection of classification algorithms.

Based on the datasets sourced, the three chosen classifications are:

* Business
* Health
* Sports

## Imports
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import string
# Data Handling and Processing
import pandas as pd
import numpy as np
import re
from scipy import interp
# Visualuzation
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format='retina'
# NLP Packages
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from joblib import dump, load
# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
# Scikit Learn packages
from sklearn.base import clone
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

"""# Load Datasets"""



text_df = pd.DataFrame(columns=['Text','Class'])
# text_df.to_csv('subject_class.csv')
text_df.head()

# Function to read the text files into one dataframe
def readfiles_to_dataframe(directory, category):
    arr = os.listdir(directory)
    strtext = ".txt"
    for textfile in arr:
        if textfile.__contains__(strtext):
            fileObject = open(directory + textfile, "r")
            data = fileObject.read()
            ouvert = pd.read_csv('news_df.csv', index_col="Unnamed: 0")
            ouvert = ouvert.append({"Class": str(category), "Text": data},ignore_index=True)
            ouvert.to_csv('news_df.csv')

# Define categories
# paths = [business_path, tech_path, arts_path]
# categories = ['business', 'tech', 'arts']


# # Call readfile function
# for path,category in zip(paths, categories):
#     readfiles_to_dataframe(path, category)

full_df = pd.read_csv('subject_class.csv')
print(full_df.shape)
full_df.head()

full_df.drop(columns=['Unnamed: 0'], inplace=True)
full_df.head()

full_df.drop(columns=['ArticleId'], inplace=True)
full_df.head()

full_df.drop(columns=['News_length'], inplace=True)
full_df.head()

full_df.drop(columns=['Text_parsed'], inplace=True)
full_df.head()

full_df.drop(columns=['Category_target'], inplace=True)
full_df.head()

"""## 1. Dataset Exploration"""

full_df['Category'].value_counts().plot(kind='bar')
plt.title('Number of News articles per Category', size=20, pad=20);

# Check for missing values
full_df.isna().sum()

"""## 2. Text Preprocessing

Here, unwanted parts of the text are removed such as special characters.
"""

def preprocess(df):
    # Remove special characters
    df['Text2'] = df['Text'].replace('\n',' ')
    df['Text2'] = df['Text2'].replace('\r',' ')

    # Remove punctuation signs and lowercase all
    df['Text2'] = df['Text2'].str.lower()
    df['Text2'] = df['Text2'].str.translate(str.maketrans('', '', string.punctuation))


    # Remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    def fwpt(each):
        tag = pos_tag([each])[0][1][0].upper()
        hash_tag = {"N": wordnet.NOUN,"R": wordnet.ADV, "V": wordnet.VERB,"J": wordnet.ADJ}
        return hash_tag.get(tag, wordnet.NOUN)


    def lematize(text):
        tokens = nltk.word_tokenize(text)
        ax = ""
        for each in tokens:
            if each not in stop_words:
                ax += lemmatizer.lemmatize(each, fwpt(each)) + " "
        return ax

    df['Text2'] = df['Text2'].apply(lematize)

import nltk
nltk.download('punkt')
nltk.download('wordnet')

preprocess(full_df)

"""### Demonstration of Preprocessing

**Original:**
"""

full_df.iloc[1]['Text']

"""**Processed:**"""

full_df.iloc[1]['Text2']

"""## 3. Train Test Split"""

X_train, X_test, y_train, y_test = train_test_split(full_df['Text2'],
                                                    full_df['Category'],
                                                    test_size=0.2,
                                                    random_state=9)

"""#### Check for acceptable category balance"""

y_train.value_counts().plot(kind='bar')
plt.title('Category Balance', size=20, pad=20);

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""## 4.  Vectorize"""

vector = TfidfVectorizer(stop_words='english',
                         ngram_range = (1,2),
                         min_df = 3,
                         max_df = 1.,
                         max_features = 10000)

"""## 5.  Construct Model"""

def fit_model(model, model_name):
    line = Pipeline([('vectorize', vector), (model_name, model)])

    output = cross_validate(line,
                            X_train,
                            y_train,
                            cv = KFold(shuffle = True,
                                       n_splits = 3,
                                       random_state = 9),
                            scoring = ('accuracy', 'f1_weighted','precision_weighted','recall_weighted'),
                            return_train_score=True)
    return output

dectree = fit_model(DecisionTreeClassifier(), 'DTree')
ridge = fit_model(RidgeClassifier(), 'Ridge')
bayes = fit_model(MultinomialNB(), 'NB')

dt = pd.DataFrame.from_dict(dectree)
rc = pd.DataFrame.from_dict(ridge)
bc = pd.DataFrame.from_dict(bayes)

l1 = [bc, rc, dt]
l2 =["NB", "Ridge", "DT"]

for each, tag in zip(l1, l2):
    each['model'] = [tag, tag, tag]

joined_output = pd.concat([bc,rc,dt])

dectree

ridge

bayes

relevant_measures = list(['test_accuracy','test_precision_weighted', 'test_recall_weighted', 'test_f1_weighted'])

dec_tree_metrics = joined_output.loc[joined_output.model == 'DT'][relevant_measures]
nb_metrics = joined_output.loc[joined_output.model == 'NB'][relevant_measures]
r_metrics = joined_output.loc[joined_output.model == 'Ridge'][relevant_measures]

"""#### Decision Tree metrics"""

dec_tree_metrics

"""#### Multinomial Naive Bayes metrics"""

nb_metrics

"""#### Ridge Classifier metrics"""

r_metrics

"""#### Average metrics"""

metrics_ = [dec_tree_metrics, nb_metrics, r_metrics]
names_ = ['Decision Tree', 'Naive Bayes', 'Ridge Classifier']

for scores, namess in zip(metrics_, names_):
    print(f'{namess} Mean Metrics:')
    print(scores.mean())
    print('  ')

"""### Selection of Model
From the metrics obtained above, we see that **Ridge Classifier** performs best. However, the **Multinomial Naive Bayes classifier** is chosen to create the final model.

This is because it **has the ability to provide probability score** for each prediction it makes, while scoring similarly to the best model.
"""

# Join training and test datasets
X = pd.concat([X_train,
               X_test])
y = pd.concat([y_train,
               y_test])

def create_and_fit(clf, x, y):
    best_clf = clf
    pipeline = Pipeline([('vectorize', vector), ('model', best_clf)])
    return pipeline.fit(x, y)

# Create model
CLASSYfier = create_and_fit(MultinomialNB(), X, y)

CLASSYfier.classes_

"""## FINAL TESTING:

The first sample text used is a tech news article about streaming services and video games.

The classifier, if appropriate, should classify this as a Sport text.
"""

input_text = 'Manchester City end interest in West Ham midfielder Declan Rice after having £90m bid rejected; Arsenal have submitted a third bid worth £105m which City are not prepared to match or surpass'
CLASSYfier.predict_proba([input_text])

CLASSYfier.predict([input_text])[0]

"""Interestingly, since the streaming services are also businesses, the model reflects this with a `0.39` probability for the business category.

## GUI
"""

import tkinter as tk
from tkinter.scrolledtext import ScrolledText


window = tk.Tk()


window.title("TEXT CLASSIFIER")
window.minsize(600,400)

text_box = ScrolledText(window)
text_box.grid(column=0, row=1, padx=5, pady=5)

def result(res, pr):
    BUSINESS = round(pr[0][0], 3)
    HEALTH = round(pr[0][1], 3)
    SPORT = round(pr[0][2], 3)

    lines = [f"Business: {BUSINESS}",f"Health: {HEALTH}", f"Sport: {SPORT}"]
    tk.messagebox.showinfo(message= f"Predicted Category: {str(res).capitalize()}" + "\n\n\n"+"\n".join(lines))

def clickMe():
    classification = tk.StringVar()
    category_,probabilities = classify_text(text_box.get("1.0",tk.END))
    result(category_, probabilities)


def classify_text(input_text):
    out = CLASSYfier.predict([input_text])[0]
    probs = CLASSYfier.predict_proba([input_text])
    return out,probs

label = tk.Label(window, text = "Enter Text to be classified")
label.grid(column = 0, row = 0)

btn = tk.Button(window, text="Classify", command=clickMe)
btn.grid(column=0, row=2)




window.mainloop()



