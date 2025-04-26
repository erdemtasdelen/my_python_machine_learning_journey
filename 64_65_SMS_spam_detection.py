import pandas as pd
#import chardet
#import matplotlib.pyplot as plt
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np




#with open("/Users/erdemtasdelen/Downloads/spam.csv","rb") as file:
#    sonuc=chardet.detect(file.read(100000))
#    print(sonuc)

data=pd.read_csv("/Users/erdemtasdelen/Downloads/spam.csv", encoding='MacRoman',names=["Etiket","Sms"], usecols=[0,1])
veri=data.copy()

#veri=veri.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

#veri=veri.rename(columns={"v1":"Etiket","v2":"Sms"})

#print(veri.groupby("Etiket").count())
#print(veri.describe())

veri=veri[veri["Etiket"] != "v1"]

veri=veri.dropna()

veri=veri.drop_duplicates()
#print(veri.describe())
#print(veri.isnull().sum())

veri["Karakter Sayısı"]=veri["Sms"].apply(len)

veri=veri.reset_index(drop=True)
#veri.hist(column="Karakter Sayısı", by="Etiket", bins=50)
#plt.show()

veri.Etiket=[1 if kod=="spam" else 0 for kod in veri.Etiket]


#mesaj=re.sub("[^a-zA-Z]", " ", veri["Sms"][0])
#print(veri["Sms"][0])
#print(mesaj)

#Ok lar... Joking wif u oni...
#Ok lar    Joking wif u oni   ---->tuyrftdekinin aynısı yaptık ama asagıda fonksiyon halinde yaptık

def harfler(cumle):
    yer=re.compile("[^a-zA-Z]")
    return re.sub(yer, " ", cumle)

#print(harfler("tuyRFT11,,???"))  #-->tuyRFT --> bu kısımda bulunan karakterlerden alfabenın icinde olmayanları bosluk ile degistirdi o yuzden boyle dondürdü.


durdurma=stopwords.words("english")


spam=[]
ham=[]
tumcumleler=[]

for i in range(len(veri["Sms"].values)):
    r1=veri["Sms"].values[i]
    r2=veri["Etiket"].values[i]

    temızcumle=[]
    cumleler=harfler(r1)
    cumleler=cumleler.lower()

    for kelimeler in cumleler.split():
        temızcumle.append(kelimeler)

        if r2==1:
            spam.append(cumleler)
        else:
            ham.append(cumleler)

    tumcumleler.append(" ".join(temızcumle))

veri["Yeni Sms"]=tumcumleler

veri=veri.drop(columns=["Sms","Karakter Sayısı"], axis=1)


cv=CountVectorizer()
x=cv.fit_transform(veri["Yeni Sms"]).toarray()

y=veri["Etiket"]
X=x

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#model=MultinomialNB()
#model.fit(X_train,y_train)
#tahmin=model.predict(X_test)


#acs=accuracy_score(y_test,tahmin)
#print(acs*100)    #-->97.30458221024259


for i in np.arange(0.0,1.1,0.1):
    model=MultinomialNB(alpha=i)
    model.fit(X_train,y_train)
    tahmin=model.predict(X_test)
    skor=accuracy_score(y_test,tahmin)
    print("Alfa {} değeri için skor: {}".format(round(i,1),round(skor*100,2)))


#Alfa 0.0 değeri için skor: 87.87
#Alfa 0.1 değeri için skor: 96.77
#Alfa 0.2 değeri için skor: 96.36
#Alfa 0.3 değeri için skor: 96.5
#Alfa 0.4 değeri için skor: 96.63
#Alfa 0.5 değeri için skor: 96.77
#Alfa 0.6 değeri için skor: 96.9
#Alfa 0.7 değeri için skor: 97.04
#Alfa 0.8 değeri için skor: 97.04
#Alfa 0.9 değeri için skor: 97.04
#Alfa 1.0 değeri için skor: 97.3

