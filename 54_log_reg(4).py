import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/red_wine_quality.csv")
veri=data.copy()

kategori=["3","4","5","6","7","8"]

oe=OrdinalEncoder(categories=[kategori])
veri["Kalite"]=oe.fit_transform(veri["quality"].values.reshape(-1,1))

print(veri[veri["quality"]==3][["quality","Kalite"]])

#      quality  Kalite
#459         3     0.0
#517         3     0.0
#690         3     0.0
#832         3     0.0
#899         3     0.0
#1299        3     0.0
#1374        3     0.0
#1469        3     0.0
#1478        3     0.0
#1505        3     0.0

veri=veri.drop(columns="quality",axis=1)
print(veri)

y=veri["Kalite"]
X=veri.drop(columns="Kalite",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

cm=confusion_matrix(y_test,tahmin)

acs=accuracy_score(y_test,tahmin)
print(acs*100)  #-->57.49999999999999

