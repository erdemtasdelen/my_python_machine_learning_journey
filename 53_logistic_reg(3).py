import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

data=pd.read_csv("/Users/erdemtasdelen/Downloads/iris_species.csv")
veri=data.copy()

veri=veri.drop(columns="Id",axis=1)

le=LabelEncoder()
veri["Species"]=le.fit_transform(veri["Species"])
#print(veri["Species"].unique())

y=veri["Species"]
X=veri.drop(columns="Species",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

cm=confusion_matrix(y_test,tahmin)
acs=accuracy_score(y_test,tahmin)

print(cm)
print(acs)


cv=cross_val_score(model,X_test,y_test,cv=10).mean()
print(cv)