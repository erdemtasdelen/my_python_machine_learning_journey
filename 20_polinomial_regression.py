#bu kodda, emlak veri seti üzerinde ev fiyatını tahmı  etmek icin polinomial regresyon modeli kurudu.
#once veriden gereksiz sutunlar atıldı, sonra ozellıkler polinomialfeatures kullanılarak 3. dereeceye genişletildi.
#boylece dogrusal model, dogrusal olmayan ilişkilere de tepki verebilir hale geldi.
#train ve test setleri olusturuuldu, model egitildi ve test verisi üzerinden tahmınler yapıldı.
#modelin basarısı r2_sscore ve mse ile ölçüldü. 



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as mt




data=pd.read_csv("/Users/erdemtasdelen/Downloads/Realestate.csv")
veri=data.copy()


#model için gereksiz görünen kolonlar siliniyor.
veri.drop(columns=["No","X1 transaction date","X5 latitude","X6 longitude"],axis=1,inplace=True)

#sutün isimlerini türkçeye cevirdik
veri=veri.rename(columns={"X2 house age":"Ev Yaşı",
                 "X3 distance to the nearest MRT station":"Metroya Uzaklık",
                 "X4 number of convenience stores":"Market Sayısı",
                 "Y house price of unit area":"Ev Fiyatı"})

#sns.pairplot(veri)
#plt.show()



y=veri["Ev Fiyatı"]
X=veri.drop(columns="Ev Fiyatı",axis=1)


#polınom donusumden geçen veriye dogrusal regresyon uygulanıyor. cünkü polynomialfeatures karmasıklastırırken model hala linear reg- sadece artik features eğri hale geldi
pol=PolynomialFeatures(degree=3)
X_pol=pol.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_pol,y,test_size=0.2,random_state=42)

pol_reg=LinearRegression()
pol_reg.fit(X_train,y_train)
tahmin=pol_reg.predict(X_test)

#r2=1'e ne kadar yakınsa o kadar iyi
#mse=0'a ne kadar yakınsa o kadar iyi
r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)

print("R2: {}  MSE: {}".format(r2,mse))
