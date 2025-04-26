import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
import sklearn.metrics as mt
import numpy as np

data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/Advertising.csv")
veri=data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

def modeltahmin(model):
    model.fit(X_train,y_train)
    tahmin=model.predict(X_test)
    #return tahmin
    r2=mt.r2_score(y_test,tahmin)
    mse=mt.mean_squared_error(y_test,tahmin)
    rmse=np.sqrt(mse)
    return [r2,rmse]

#print(modeltahmin(LinearRegression()))
#print(modeltahmin(LinearRegression()))

modeller=[LinearRegression(),Ridge(),Lasso(),ElasticNet(),SVR(),DecisionTreeRegressor(random_state=0),BaggingRegressor(random_state=0),RandomForestRegressor(random_state=0)]

ad=["Linear Model","Ridge Model","Lasso Model","ElasticNet Model","SVR Model","Karar Ağacı Modeli","Bag Model","Random Forest Model"]

sonuc=[]

for i in modeller:
    sonuc.append(modeltahmin(i))
#print(sonuc)


df=pd.DataFrame(ad,columns=["Model Adı"])
#print(df)

df2=pd.DataFrame(sonuc,columns=["R2","RMSE"])

df=df.join(df2)
print(df)


#             Model Adı        R2      RMSE
#0         Linear Model  0.904906  1.767425
#1          Ridge Model  0.904904  1.767446
#2          Lasso Model  0.903843  1.777275
#3     ElasticNet Model  0.904189  1.774073
#4            SVR Model  0.818989  2.438466
#5   Karar Ağacı Modeli  0.936574  1.443433
#6            Bag Model  0.965222  1.068848 -->bu da iyi sayılır
#7  Random Forest Model  0.978299  0.844316 -->digerlerine kıyasla cok iyi

