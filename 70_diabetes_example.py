import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/diabetes_dataset.csv")
veri=data.copy()
print(veri.info())

y=veri["Outcome"]
X=veri.drop(columns="Outcome",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

def modeller(model):
    model.fit(X_train,y_train)
    tahmin=model.predict(X_test)
    skor=accuracy_score(y_test,tahmin)
    return round(skor*100,2)

#print(modeller(DecisionTreeClassifier()))  #-->74.68

models=[]

models.append(("Log Regression", LogisticRegression(random_state=0)))
models.append(("KNN", KNeighborsClassifier()))
models.append(("SVC", SVC(random_state=0)))
models.append(("Bayes", GaussianNB()))
models.append(("Decision Tree", DecisionTreeClassifier(random_state=0)))


modelad=[]
basarı=[]

for i in models:
    modelad.append(i[0])
    basarı.append(modeller(i[1]))

#print(modelad)   #-->['Log Regression', 'KNN', 'SVC', 'Bayes', 'Decision Tree']
#print(basarı)    #-->[LogisticRegression(random_state=0), KNeighborsClassifier(), SVC(random_state=0), GaussianNB(), DecisionTreeClassifier(random_state=0)]
                 #--> [75.32, 69.48, 73.38, 76.62, 74.68]


a=list(zip(modelad,basarı))
sonuc=pd.DataFrame(a,columns=["Model","Skor"])
print(sonuc)

#            Model   Skor
#0  Log Regression  75.32
#1             KNN  69.48
#2             SVC  73.38
#3           Bayes  76.62
#4   Decision Tree  74.68


