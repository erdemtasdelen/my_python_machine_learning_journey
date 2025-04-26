#ÖZET#
#Bu projede SVR (Support Vector Regression) ile THYAO hissesinin kısa vadeli fiyat hareketlerini tahmin etmek için veri Yahoo Finance’tan çekildi. 
#Girdi olarak gün numarası, çıktı olarak açılış fiyatı alındı. Veriler standartlaştırıldıktan sonra RBF kernel ile SVR modeli kuruldu. 
#R² ve RMSE gibi metriklerle performans ölçüldü. Daha sonra GridSearchCV ile farklı kernel, C ve gamma kombinasyonları denenerek en iyi hiperparametreler seçildi. 
#Sonuçlar hem görselleştirildi hem de modelin tuning ile iyileştirildiği ispatlandı.



import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV



     #VERİ ÇEKME VE HAZIRLAMA#
#--> yahoo financetan thyao hisse senedi verisi cektim
#--> tarih kolonundan sadece gün bilgisini ali day olarak kullandım
data=yf.download("THYAO.IS",start="2022-08-01",end="2022-09-01")
veri=data.copy()
veri=veri.reset_index()
veri["Day"]=veri["Date"].astype(str).str.split("-").str[2]

y=veri["Open"]
X=veri["Day"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)

scy=StandardScaler()
scx=StandardScaler()


#StandartScaler ile veriler normalize ediliyor-->model daha iyi öğrenyiyor. SVR ölçek hassastir
X=scx.fit_transform(X)
y=scy.fit_transform(y)



#RBF(Gaussian)kernel kullandım-->nonlinear ilişki ogrenmek icin en iyisi
#C=10000-->yüksek ceza-->overfitting eğilimi olabilir ama kısa vadede etkili
svrrbf=SVR(kernel="rbf",C=10000)
svrrbf.fit(X,y)
tahminrbf=svrrbf.predict(X)

#svrlin=SVR(kernel="linear")
#svrlin.fit(X,y)
#tahminlin=svrlin.predict(X)

#svrpoly=SVR(kernel="poly",degree=2)
#svrpoly.fit(X,y)
#tahminpoly=svrpoly.predict(X)

r2=mt.r2_score(y,tahminrbf)
mse=mt.mean_squared_error(y,tahminrbf)
rmse=np.sqrt(mse)

print("R2: {}  RMSE: {} ".format(r2,rmse))




parametreler={"C":[1,10,100,1000,10000],"gamma":[1,0.1,0.001],"kernel":["rbf","linear","poly"]}

tuning=GridSearchCV(estimator=SVR(),param_grid=parametreler,cv=10)
tuning.fit(X,y)
print(tuning.best_params_)

#-->{'C': 10000, 'gamma': 0.001, 'kernel': 'rbf'}
#-->aralarında C icin en optimal deger 10000, gamme icin 0.001, kernel icin de rbf'mis onu döndürdü


plt.scatter(X,y,color="red")
plt.plot(X,tahminrbf,color="green",label="RBF Model")
#plt.plot(X,tahminlin,color="blue",label="Linear Model")
#plt.plot(X,tahminpoly,color="pink",label="Poly Model")
plt.legend()
plt.show()
