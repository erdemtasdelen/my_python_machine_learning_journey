             #KODU ÖZETLEYELİM#
#bu kod, ABD konut fiyatlarını acıklamak icin 4 farklı regresyon modelini(linear,ridge,lasso,elasticnet)karsılastırrıyorr.
#her modelin test setindeki performansını RMSE VE R2 ile ölçülüyor, ardından capraaz dogrulama ile genelleme gücü(dogrulama skoru) öğreniliyor.
#sonuclar, biçimlendirilmis bir dataframe icinde kıyaslanıyor.. pd.options.display.float_format gibi ayrlarla cıktının okunabilirligi artırılıyor
#bu yapı sayesınde hangi modelin daha dogru, daha tutarlı ve daha dengeli oldugu sistematik bicimde degerlendirilebiliyor


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import numpy as np
#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor


#VERİ OKUMA VE ÖN İŞLEME 
#csv dosyası okunuyor.
#orijinali bozulmasın diye kopyalanıyor
#adress sütunu işe yaramadıgı icin atılıyor.
data=pd.read_csv("/Users/erdemtasdelen/Downloads/USA_Housing.csv")
veri=data.copy()

veri=veri.drop(columns="Address",axis=1)


#sns.pairplot(veri)
#plt.show()


#kor=veri.corr()
#sns.heatmap(kor,annot=True)
#plt.show()


y=veri["Price"]
X=veri.drop(columns="Price",axis=1)

#sabit=sm.add_constant(X)

#vif=pd.DataFrame()
#vif["Değişkenler"]=X.columns
#vif["VIF"]=[variance_inflation_factor(sabit,i+1) for i in range(X.shape[1])]
#print(vif)


          #SONUCLAR#
#                   Değişkenler       VIF
#0              Avg. Area Income  1.001159
#1           Avg. Area House Age  1.000577
#2     Avg. Area Number of Rooms  1.273535
#3  Avg. Area Number of Bedrooms  1.274413
#4               Area Population  1.001266


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


        #FONKSİYON TANIMLAMALARI#
#caprazdog--> capraz dogrulama ile genel basarı
#10 katlı capraz dogrulama ile modelin genel basarısı ölçülüyor
#bu, modelin sadece test setine degil, tüm veriye karsı ortalama basarısını gösterir
#basarı--> RMSE ve R2 ölçen fonksiyon
def caprazdog(model):
    dogruluk=cross_val_score(model,X,y,cv=10)
    return dogruluk.mean()


def basarı(gercek,tahmin):
    mse=mt.mean_squared_error(gercek,tahmin)
    rmse=np.sqrt(mse)  #-->karekök alınarak RMSE hesaplanıyor
    r2=mt.r2_score(gercek,tahmin)
    return [rmse,r2]



   #MODEL EĞİTİM VE TAHMİNLERİ#
#4 farklı regresyon modeli eğitiliyor
#hepsinin tahminleri alınarak test verisinde performansları ölçülecek
lin_model=LinearRegression()
lin_model.fit(X_train,y_train)
lin_tahmin=lin_model.predict(X_test)

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)
ridge_tahmin=ridge_model.predict(X_test)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_tahmin=lasso_model.predict(X_test)

elas_model=ElasticNet(alpha=0.1)
elas_model.fit(X_train,y_train)
elas_tahmin=elas_model.predict(X_test)


#print(basarı(y_test,lin_tahmin))
#-->ilk deger rmse, ikinci deger r2 degeridir
#np.float64(100444.06055557792), 0.9179971706834444]




     #SONUCLARIN TABLO HALİNDE TOPARLANMASI#
sonuclar=[["Linear Model",basarı(y_test,lin_tahmin)[0],basarı(y_test,lin_tahmin)[1],caprazdog(lin_model)],
["Ridge Model",basarı(y_test,ridge_tahmin)[0],basarı(y_test,ridge_tahmin)[1],caprazdog(ridge_model)],
["Lasso Model",basarı(y_test,lasso_tahmin)[0],basarı(y_test,lasso_tahmin)[1],caprazdog(lasso_model)],
["Elastic Net Model",basarı(y_test,elas_tahmin)[0],basarı(y_test,elas_tahmin)[1],caprazdog(elas_model)]]


    #ÇIKTI FORMATI AYARLNIYOR#
#tüm ondalıklı cıktılar 4 basamakla ve binlik ayırıcıyla görünür
pd.options.display.float_format='{:,.4f}'.format


     #SONUÇLARIN DATAFRAME'E DÖNÜŞTÜRÜLÜP GÖSTERİLMES#
sonuclar=pd.DataFrame(sonuclar,columns=["Model","RMSE","R2","Dogrulama"])
print(sonuclar)

           #SONUCLAR#
#               Model         RMSE     R2  Dogrulama
#0       Linear Model 100,444.0606 0.9180     0.9174   -->en ideal model
#1        Ridge Model 100,444.4114 0.9180     0.9174
#2        Lasso Model 100,444.0666 0.9180     0.9174
#3  Elastic Net Model 101,606.4311 0.9161     0.9165   -->elastic net hem ridge hem lassoyu beraber kullandıgından daha düsük acıklama yetenegine sahip bir model olusturur

#RMSE->düşük olanı makbul
#R2->1'e en yakın olanı makbül ---> r2, modelin acıklama gücüdür.
#Validation->yüksek olanı makbul