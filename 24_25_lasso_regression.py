#Lasso Regresyon Analizi: klaasık coklu reg. analizinde X'lerin Y üzerindeki etkisinin anlamlı olup olmadıgını test ederiz.
#kurulan model sonucunda bagımsız degisken anlamsız bile bulunsa, matematiksel acıdan modelde yine yer almaktadır.
#lasso reg. analizi ise anlamsız degiskenleri "modelden dışlama" ozelligi olan muazzam bir yaklasımdır.
#Lasso reg = (Yi-Yi(predict))^^2 + lambda (x) |Beta|





import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LassoCV
#import sklearn.metrics as mt

df=load_wine()

data=pd.DataFrame(df.data,columns=df.feature_names)

veri=data.copy()

veri["WINE"]=df.target


y=veri["WINE"]
X=veri.drop(columns="WINE",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#ridge_model=Ridge(alpha=0.1)
#ridge_model.fit(X_train,y_train)
#tahmin=ridge_model.predict(X_test)

#print(ridge_model.score(X_train,y_train))
#print(ridge_model.score(X_test,y_test))

lasso_model=Lasso(alpha=0.01)  #-->alpha degerini elimizden geldigince küçük tutmak modelimiz icin gereklidir.
lasso_model.fit(X_train,y_train)

#print(lasso_model.score(X_train,y_train))
#print(lasso_model.score(X_test,y_test))

print(lasso_model.score(X_train,y_train))
print(lasso_model.score(X_test,y_test))


lamb=LassoCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_


lasso_model2=Lasso(alpha=lamb)  
lasso_model2.fit(X_train,y_train)

print(lasso_model2.score(X_train,y_train))
print(lasso_model2.score(X_test,y_test))


    #SONUCLAR#
#0.8915281102080611-->lasso1 train
#0.8825576304658078-->lasso1 test
#0.8223219136017651-->lasso2 train
#0.8171860269099442-->lasso2 test

