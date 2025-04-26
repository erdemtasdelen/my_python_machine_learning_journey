
import pandas as pd     #--> pandas veri analizi icin temel kütüphanedir.read_excel(),datafram,iloc,copy() gibi fonksiyonlar pandas'tan gelir 
from sklearn.impute import SimpleImputer  #-->simpleimputer, eksik verileri doldurmak icin kullanılır. NaN değeleri mean,medyan ve en sık degerlerle doldurulur
import numpy as np    #--> numpy sayısal islemler icin kullanılır.
from sklearn.model_selection import train_test_split   #--> veriyi eğitim ve test olarak ayırır. train_test_split(), supervised learningin temelidir
from sklearn.linear_model import LinearRegression  #--> linearregression, sklearndeki basit dogrusal regresyon modelidir
import sklearn.metrics as mt   #--> r2_score,mean_squared_error,mean_absolute_error gibi performanc ölçümleri buradan gelir

#VERİ OKUMA VE KOPYALAMA
#read_excel(), excel dosyasını dataframe'e ceviri
#copy(), orijinal veriyi korumak icin yedegini alıyoruz. üzerinde calısacagimiz versiyonu : "veri" oluyor
data=pd.read_excel("/Users/erdemtasdelen/Desktop/python_lessons/Advertising2.xlsx")
veri=data.copy()


#EKSİK VERİYİ ORTALAMA İLE DOLDURMA
#simpleimputer: eksik verileri belirli bir stratejiye göre dolduracak
#fit(): hangi kolonlarda eksik var, hangi ortalama hesaplanacak -->öğreniyor.
#transform(): eksikleri ortalamalarla dolduruyor.
#iloc[:,:]: tüm satır ve sutünları etkiler.
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(veri)
veri.iloc[:,:]=imputer.transform(veri)

#print(veri.iloc[4])

y=veri["Sales"]
X=veri[["TV","Radio"]]

#VERİYİ TRAİN TEST OLARAK AYIRMA
#eğitim: modelin öğrenecegi veri
#test: modelin sınanacağı yer 
#random_state=42: aynı sonucu tekrar tekrar almak icin
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#MODEL OLUSTURMA VE TAHMİN
#fit(...): modeli eğitiyoruz, yani X_traindeki örüntüleri y_traine'e göre öğreniyor
#predict(): X_test verisine bakarak tahmin yapıyor
lr=LinearRegression()  #--> model nesnesi yaratıldı
lr.fit(X_train, y_train)
tahmin=lr.predict(X_test)


#MODEL PERFORMANS ÖLÇÜMÜ
#r2_score: determinasyon katsayısı,1'e ne kadar yakınsa o kadar iyi
#mse: ortalama karesel hata. küçük olması beklenir. aykırılara duyarlıdır.
#mae: ortalama mutlak hata. MSEye göre daha saglamdır, ama yorumlaması zordur.
r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
mae=mt.mean_absolute_error(y_test,tahmin)

print(r2,mse,mae)
