# %%
import numpy as np
import pandas as pd

# %%
message=[line.rstrip() for line in open("SMSSpamCollection")]

# %%
message

# %%
message[0]

# %%
message=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','text'])

# %%
message.head()

# %%
message.iloc[0]

# %% [markdown]
# Cleaning the data

# %%
import string

# %%
string.punctuation

# %%
import nltk

# %%
from nltk.corpus import stopwords

# %%
from nltk.stem.porter import PorterStemmer

# %%
stemmer=PorterStemmer()

# %%
def clean(text):
    text=text.lower()
    text=''.join([c for c in text if c not in string.punctuation])
    words=text.split()
    words=[stemmer.stem(w) for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# %%
message['new_text']=message['text'].apply(clean)

# %%
message

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
from sklearn.model_selection import train_test_split

# %%
from sklearn.pipeline import Pipeline

# %%
from sklearn.model_selection import GridSearchCV

# %%
from sklearn.linear_model import LogisticRegression

# %%
message.dropna(subset='label',inplace=True)

# %%
message.isna().sum()

# %%
message['label']=message['label'].map({'spam':0,'ham':1})

# %%
message

# %%
message['label'].isna().sum()

# %%
X=message['new_text']

# %%
y=message['label']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %% [markdown]
# upsample the spam class

# %%
from sklearn.utils import resample
import pandas as pd

# %%
 #Combine into a single DataFrame
df = pd.concat([X_train, y_train], axis=1)

# %%
# Separate majority and minority
ham = df[df['label'] == 1]
spam = df[df['label'] == 0]

# %%
# Upsample spam
spam_upsampled = resample(spam,
                          replace=True,
                          n_samples=len(ham),
                          random_state=42)

# %%
# Combine back
upsampled = pd.concat([ham, spam_upsampled])
X_train_balanced = upsampled['new_text']
y_train_balanced = upsampled['label']

# %% [markdown]
# Logistic Regression

# %%
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# %%
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2),(1,3)],
    'tfidf__max_df': [0.9, 1.0],
    'tfidf__min_df': [1, 2],
    'clf__C': [0.1, 1, 10]
}

# %%
from sklearn.metrics import classification_report,confusion_matrix

# %%
from sklearn.model_selection import GridSearchCV

# %%
grid=GridSearchCV(pipeline,param_grid,scoring='f1_macro',cv=5,n_jobs=-1)

# %%
grid.fit(X_train_balanced,y_train_balanced)

# %%
print("Best parameters:",grid.best_params_)

# %%
print("Best F1 Score (CV):", grid.best_score_)

# %%
best_model=grid.best_estimator_

# %%
pred=best_model.predict(X_test)

# %%
print(classification_report(y_test,pred))

# %%
print(confusion_matrix(y_test,pred))

# %% [markdown]
# Naive Bayes Classifier

# %%
from sklearn.naive_bayes import MultinomialNB

# %%
pipeline=Pipeline([
    ('tfid',TfidfVectorizer()),
    ('clf',MultinomialNB())
])

# %%
param_grid_nb = {
    'tfid__ngram_range': [(1, 1), (1, 2)],
    'tfid__min_df': [1, 2, 5],
    'tfid__max_df': [0.8, 0.9, 1.0],
    'tfid__sublinear_tf':[True,False],
    'clf__alpha': [0.01, 0.1,0.5, 1, 5, 10]  # Alpha is smoothing parameter in Naive Bayes
}

# %%
grid=GridSearchCV(estimator=pipeline,param_grid=param_grid_nb,scoring='f1_macro',n_jobs=-1,cv=5)

# %%
grid.fit(X_train_balanced,y_train_balanced)

# %%
print(grid.best_estimator_)

# %%
print(grid.best_params_)

# %%
print(grid.best_score_)

# %%
bm=grid.best_estimator_

# %%
pred=bm.predict(X_test)

# %%
print(classification_report(y_test,pred))

# %%
print(confusion_matrix(y_test,pred))

# %%
message['label'].value_counts()

# %% [markdown]
# Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
pipeline=Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('clf',RandomForestClassifier())
])

# %%
param_grid = {
    'tfidf__max_df': [0.7, 0.9],
    'tfidf__min_df': [2, 5],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_features': [1000, 3000, None],

    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 30],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__class_weight': [None, 'balanced'],  # Important for imbalanced data
    'clf__max_features': ['sqrt']
}

# %%
grid=GridSearchCV(estimator=pipeline,n_jobs=-1,param_grid=param_grid,scoring='f1_macro',cv=5)

# %%
grid.fit(X_train_balanced,y_train_balanced)

# %%
print("Best parameters",grid.best_estimator_)

# %%
bmr=grid.best_estimator_

# %%
print(grid.best_score_)

# %%
pred=bmr.predict(X_test)

# %%
print(classification_report(y_test,pred))

# %%
print(confusion_matrix(y_test,pred))

# %%
import joblib

# %%
joblib.dump(best_model,'spam_classifier.pkl')

# %%