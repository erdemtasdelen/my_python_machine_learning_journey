#hiper parametre optimizasyonu: train test verilerini optimal sekilde bölmek
#parametreleri biz degil model tahmin eder ancak hiper parametreleri biz seçeriz. for ex: k-fold cross validation...

#bu kod, bir reklam veri seti üzerinde linear regression modelini hem klasık train-test ayrımı hem de 5 katlı capraz sorgulama K-Fold yontemiyle test eddiyor
#ilk olarak veriyi temizledik, bagımlı, bagımsız degiskenleri ayırıdk, modeli egittik ve performansını ölçtük.
#daha sonra k-fold yontemıylr model alt kumeler uzerınde tekrar tekrar egitilerek her seferinde train ve test basarıları(r2 ve MSE)hesaplanıyor
#bu sayede modelin kararlılıgı ve genelleme gücü daha dogru degerlendiriliyor.


import pandas as pd
from sklearn.model_selection import train_test_split,KFold #-->KFold, capraz dogrulama icin kullanılır.
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt

data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/Advertising.csv")
veri=data.copy()
veri.drop(columns=["Unnamed: 0"],axis=1,inplace=True)

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#dogrusal regresyon modeli tanımlanıyor ve egitim verisiyle egitiliyor.
lr=LinearRegression()
model=lr.fit(X_train,y_train)


#skor adında bir fonksiyon tanımladık
#bu fonksiyon modele hem train hem test verisini veriyor. r2 ve MSE hesaplıyor, sonucları liste halinde döndürüyor.
def skor(model,x_train,x_test,y_train,y_test):
    egitimtahmin=model.predict(x_train)
    testtahmin=model.predict(x_test)
    

    r2_egitim=mt.r2_score(y_train,egitimtahmin)
    r2_test=mt.r2_score(y_test,testtahmin)

    mse_egitim=mt.mean_squared_error(y_train,egitimtahmin)
    mse_test=mt.mean_squared_error(y_test,testtahmin)

    return[r2_egitim,r2_test,mse_egitim,mse_test]

sonuc1=skor(model=lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)


print("Eğitim R2={}  Eğitim MSE={}".format(sonuc1[0],sonuc1[2]))
print("Test R2={}  Test MSE={}".format(sonuc1[1],sonuc1[3]))

#K-Fold çapraz dogrulama icin yeni bir model tanımlanıyor. 5 parçalı bölme(k=5) yapılıyor
lr_cv=LinearRegression()
k=5
iterasyon=1
cv=KFold(n_splits=k)


#veriler her iterasyonda yeniden bölünüyor. model her katmanda eğitiliyor, skor fonksiyonu ile train ve test performansı ölçülüyor. her iterasyonun sonucları yazdırılıyor.
for egitimindex,testindex in cv.split(X):
    X_train,X_test=X.loc[egitimindex],X.loc[testindex]
    y_train,y_test=y.loc[egitimindex],y.loc[testindex]
    lr_cv.fit(X_train,y_train)
    
    sonuc2=skor(model=lr_cv,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)

    print("İterasyon:{}".format(iterasyon))
    print("Eğitim R2={}  Eğitim MSE={}".format(sonuc2[0],sonuc2[2]))
    print("Test R2={}  Test MSE={}".format(sonuc2[1],sonuc2[3]))
    iterasyon +=1




#Eğitim R2=0.8957008271017817  Eğitim MSE=2.7051294230814142
#Test R2=0.899438024100912  Test MSE=3.174097353976105
#İterasyon:1
#Eğitim R2=0.9010130247585828  Eğitim MSE=2.7115931715887234
#Test R2=0.8786519804831341  Test MSE=3.1365399007617043
#İterasyon:2
#Eğitim R2=0.8903959783952622  Eğitim MSE=2.8896961578499276
#Test R2=0.9176321165614463  Test MSE=2.4256677581593866
#İterasyon:3
#Eğitim R2=0.8896931584883978  Eğitim MSE=3.104396076662707
#Test R2=0.9293303235799653  Test MSE=1.5852250798740992
#İterasyon:4
#Eğitim R2=0.9145880146193406  Eğitim MSE=2.241641526638164
#Test R2=0.8144390391722338  Test MSE=5.426155060429457
#İterasyon:5
#Eğitim R2=0.8961523241120161  Eğitim MSE=2.82179249487708
#Test R2=0.8954782879224387  Test MSE=2.791145186276396