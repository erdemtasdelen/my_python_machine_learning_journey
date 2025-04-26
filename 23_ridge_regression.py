#bu calısmada, klasık dogrusal regresyon modeliyle ridge regression modeli karsılastırıldı.
#ridge modeli alpha hiperparametresi aracılıgıyla katsayıları bastırarak modelin asırı uyum yapmasını engeller
#baslangıcta alpha degerş elle verildi ve modelin test basarımı ölçüldü
#daha sonra ridgeCV kullanılarak, logaritimik aralıkta denenmis 100 farklı alpha degeri arasından cross validation ile en iyi hiperparametre otomatik olarak belırlendi
#bu yaklasım, modelin regularizasyon gücünü optimize ederek hem genelleme yetenegın artırır hem de katsayıların kontrolsuz büyümesini engeller.



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
import sklearn.metrics as mt
import numpy as np


data=pd.read_excel("/Users/erdemtasdelen/Downloads/coklu_dogrusal_baglantı.xlsx")
veri=data.copy()



#korelasyon analizi için ısı haritası... burada multicollinearity(x'lerin birbirleriyle ilişkisi)analiz edilebilir
#sns.heatmap(veri.corr(),annot=True)
#plt.show()

y=veri["Y"]
X=veri.drop(columns="Y",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin=lr.predict(X_test)

r2=mt.r2_score(y_test,tahmin)

print("R2: {}".format(r2))

ridge_model=Ridge(alpha=201.85086292982749)
ridge_model.fit(X_train,y_train)
tahmin2=ridge_model.predict(X_test)

r2rid=mt.r2_score(y_test,tahmin2)

print("R2 rid: {}".format(r2rid))

lambdalar=10**np.linspace(10,-2,100)*0.5   #--> bu deger bize lambda degerleri üretiyor yani sayı üretiyor. lambda aralıgı olusturduk



#RidgeCV modeli ile cross validation yapılarak en iyi alpha degeri otomatik olarak seciliyor.
ridge_cv=RidgeCV(alphas=lambdalar,scoring="r2")
ridge_cv.fit(X_train,y_train)
#print(ridge_cv.alpha_)---> alphanın optimum degerini bulduk.





