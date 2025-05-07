import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: Display full width columns for preview
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 100)

df=pd.read_csv('spam.tsv', sep="\t")
df.columns = df.columns.str.strip()

hamDf=df[df['label'] == 'ham']
spamDf=df[df['label'] == 'spam']
hamDf=hamDf.sample(spamDf.shape[0])

finalDf = pd.concat([hamDf, spamDf], ignore_index=True)
# print(finalDf.shape)

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(finalDf['message'], finalDf['label'], test_size=0.2, random_state=0, shuffle=True, stratify=finalDf['label'])

#classifiers creation using pipeline
# hardware ko extreme use karne ke liye pipeline ka use kiya gaya hai
# pipeline ka use karne se code ko modularity milti hai aur readability bhi badhti hai
#2 stage pipeline 
model=Pipeline([('tfidf',TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=100,n_jobs=-1))])

model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

# print(confusion_matrix(Y_test,Y_pred))
# print(classification_report(Y_test,Y_pred))

print(accuracy_score(Y_test,Y_pred))
# print(model.predict(['I am a good']))

joblib.dump(model, 'random_forest_model1.pkl')

model1 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', SVC(C=1000, gamma='auto'))  # Corrected: C instead of c
])
model1.fit(X_train, Y_train)
Y_pred = model1.predict(X_test)
print(accuracy_score(Y_test, Y_pred))

joblib.dump(model1, 'svm_model1.pkl') #more accuracy