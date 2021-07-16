from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
import regex as re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk
import warnings 
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
from csv import writer

from django.contrib import messages
# Create your views here.
 
def indexView(request):
    return render(request,'home.html')

def homeView(request):
    return render(request,'home.html') 

def aboutView(request):
    return render(request,'aboutus.html')           

@login_required(login_url='/login')
def dashboardView(request):
    messages.success(request,'Logged In Succesfully!')
    return render(request,'dashboard.html') 

def registerView(request): 
    if request.method == "POST" :
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,'Account Succesfully Created!')
            return redirect('login_url')
    else:
        form = UserCreationForm()
        
    return render(request,'registration/register.html',{'form' :form})

def remove_pattern(text,pattern):
        
        # re.findall() finds the pattern i.e @user and puts it in a list for further task
        r = re.findall(pattern,text)
        
        # re.sub() removes @user from the sentences in the dataset
        for i in r:
            text = re.sub(i,"",text)
        
        return text



@login_required(login_url='/login')
def resultView(request):
    
    print("Inside views")
    if(request.method=='POST'):
        tweet=request.POST['tweet']
    print(tweet)
    
    train = pd.read_csv("C:/Users/Dharmesh/Downloads/projects-20210424T054957Z-001/projects/train.csv")

    train_original=train.copy()
   
    df2 = {'ID': len(train.index)+1, 'label': 100, 'tweet': tweet,'Tidy_Tweets':tweet}
    train = train.append(df2, ignore_index = True)

    train['Tidy_Tweets'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")

    train['Tidy_Tweets'] = train['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

    train['Tidy_Tweets'] = train['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    token_tweet = train['Tidy_Tweets'].apply(lambda x: x.split())

    ps = PorterStemmer()

    token_tweet = token_tweet.apply(lambda x: [ps.stem(i) for i in x])

    tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

    tfidf_matrix=tfidf.fit_transform(train['Tidy_Tweets'])

    df_tfidf = pd.DataFrame(tfidf_matrix.todense())

    length_df=len(df_tfidf.index)

    train_tfidf=tfidf_matrix[:length_df-1]
    test_tfidf=tfidf_matrix[length_df-1:]

    train_tfidf.todense()
    test_tfidf.todense()

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
        pred='YOUR TWEET MIGHT CONTAIN RACIST/SEXIST COMMENTS OR HATE SPEECH AND THUS IT MIGHT BE NEGATIVE.'
       
    else:
        pred='YOUR TWEET DOES NOT CONTAIN RACIST/SEXIST COMMENTS OR HATE SPEECH AND THUS IT MIGHT BE POSITIVE.'
        
    context={'tweet':tweet, 'prediction':pred, 'flag':test_pred_int[0]}

    List=[len(train.index),test_pred_int[0],tweet]
  
    with open("C:/Users/Dharmesh/Downloads/projects-20210424T054957Z-001/projects/train.csv","a",newline="") as file:
        writer = csv.writer(file) 
        writer.writerow(List)
    file.close()        
    return render(request,'results.html',context)      
