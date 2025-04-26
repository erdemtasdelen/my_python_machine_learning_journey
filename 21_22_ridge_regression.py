#rigde regression: overfittinge dirençli bir yontemdir.
#multicollinearity(X'lerin birbirine cok benzemesi(gelir-harcanabilir gelir)ne dirençli bir yontemdir)
#aykırı degerlere dirençlere dirençli bir yapı
#anlamsız parametre yapılarını modelden dıslamak yerıne katsayılarını 0'a yaklastırarak model icinde anlamsız kılar. ama model dısına atmıyor yapıyı bu onemli
#Lambda parametresi, bizim hazırlayabilecegimiz bir hiper parametredir.
#RİDGE MODELİN ANAYAPISI, "LAMBDALAR YUKARI CIKTIKCA KATSAYILAR SIFIRA YAKLASIR"


#bu uygulamada dogrusal regresyon modeliyle(L2 regularizasyonlu)regresyon modeli karsılastırıldı.
#ridge, alpha hiperparametresi ile model katsayılarını bastırarak asırı uyumu engellemeyi amaçlar.
#aynı train/test verisiyile her iki model egitildi ve bassarılar r2/mse ile kıyaslandı
#ardından farklı alpha degerleriyle ridge modeller kurulp katsayı degisimleri izlendi.
#grafik yardımıyla alpha arttıkca modelın basıtlestigi(katsayıların sıfıra yaklastıgı) gosterildi
#bu, model karmasıklıgını cezalandırarak genelleme gücünü artırmak isteyenler icin güclü bir teknik



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/Advertising.csv")
veri=data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin=lr.predict(X_test)

r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)

print("R2: {}  MSE: {}".format(r2,mse))



#aynı veri ile ridge regression modeli kuruluyor, alpha=150--> ceza katsayısı(en kadar yüksekse, katsayılar o kadar bastırılır)
ridge_model=Ridge(alpha=150)        #-->alpha hiper parametree, yani bizim secebildigimiz bir parametredir.
ridge_model.fit(X_train,y_train)
tahmin2=ridge_model.predict(X_test)


#ridge modelinin test performansı da ölçülüyor. bu sayede klasık regresyon ıle kıyaslama yapılabılıyor
r2rid=mt.r2_score(y_test,tahmin2)
mserid=mt.mean_squared_error(y_test,tahmin2)

print("R2 rid: {}  MSE rid: {}".format(r2rid,mserid))



#alpha(lambda) degerleri icin genıs bir aralık olusturuluyor. büyük ceza--> kücük ceza olacak sekılde logaritmik ölçekle
katsayılar=[]
lambdalar=10**np.linspace(10,-2,100)*0.5



#her alpha icin yeni ridge modeli eğitiliyor ve katsayılar kaydediliyor. boylece alpha büyüdükçe regresyon katsayılarının nasıl kuculdugu ızlenecek
for i in lambdalar:
    ridmodel=Ridge(alpha=i)
    ridmodel.fit(X_train,y_train)
    katsayılar.append(ridmodel.coef_)

#katsayılarım lambda(alpha) ile nasıl degistigi logaritmik eksende ciziliyor. bu grafik:regularizasyonun etkisini gorsel olarak verir. katsayılar yüksek cezayla sıfıra yaklasır
ax=plt.gca()
ax.plot(lambdalar,katsayılar)
ax.set_xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Katsayılar")
plt.show()


