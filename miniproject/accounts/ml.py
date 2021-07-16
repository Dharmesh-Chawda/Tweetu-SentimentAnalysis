import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 



train = pd.read_csv("train.csv")

train_original=train.copy()

tweets=input("Enter your Tweet:")
df2 = {'ID': len(train.index)+1, 'label': 100, 'tweet': tweets,'Tidy_Tweets':tweets}
train = train.append(df2, ignore_index = True)

def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text

train['Tidy_Tweets'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")



train['Tidy_Tweets'] = train['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

train['Tidy_Tweets'] = train['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))



token_tweet = train['Tidy_Tweets'].apply(lambda x: x.split())


from nltk import PorterStemmer

ps = PorterStemmer()

token_tweet = token_tweet.apply(lambda x: [ps.stem(i) for i in x])





from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(train['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

length_df=len(df_tfidf.index)




train_tfidf=tfidf_matrix[:length_df-1]
test_tfidf=tfidf_matrix[length_df-1:]

train_tfidf.todense()
test_tfidf.todense()

from sklearn.linear_model import LogisticRegression

Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')

x_train=train_tfidf
y_train=train_original['label'][:]


Log_Reg.fit(x_train,y_train)

prediction_tfidf = Log_Reg.predict_proba(test_tfidf)

print(prediction_tfidf)

test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1] >= 0.3

test_pred_int = test_pred_int.astype(np.int)

print(test_pred_int[0])

if test_pred_int[0]==1:
    print("YOUR TWEET MIGHT CONTAIN RACIST/SEXIST COMMENTS OR HATE SPEECH AND THUS IT MIGHT BE NEGATIVE")
else:
    print("YOUR TWEET DOES NOT CONTAIN RACIST/SEXIST COMMENTS OR HATE SPEECH AND THUS IT MIGHT BE POSITIVE")

