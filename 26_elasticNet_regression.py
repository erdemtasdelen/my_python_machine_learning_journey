#elasticNet: ridge ile katsayıları cezalandırırken, Lasso yapısı ile değişken seçer. 
#Lasso katsayıları 0'a indirgeyerek yapıyı dısarı atmayı saglar. Burada cezalandırma yapar. Bu durumda o değişken modelin dışına atılmıs olur.


#bu derste ridge,lasso,elasticnet regresyon modelleri meme kanseri veri seti üzerinde karsılastırıldı.
#ilk olarak sabit alpha=0.1 degeriyle üç model de egitildi ve R2/MSE performansları degerlendirildi
#ardından elasticNetCV kullanılarak 10 katlı capraz dogrulama ile en uygun alpha degeri otomatik olarak belirlendi
#bu alpha degeriyle yeniden egitilen modelin performansı ölçüldü ve regularizasyonun etkisi dogrudan gozlemlendi
#bu süreç, hem asırı uyumu(overfitting) azaltmak hem de gereksiz degiskenleri baskılayarak sade bir model kurmak acısından cok güclü bir ML stratejisi ortaya koyar.





import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
import sklearn.metrics as mt


df=load_breast_cancer()

data=pd.DataFrame(df.data,columns=df.feature_names)

veri=data.copy()

y=df.target
X=veri

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#L2 bastırır.
ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

#L1 bazi katsayıları sıfırlar.
lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)

#L1+L2 karması
elas_model=ElasticNet(alpha=0.1)
elas_model.fit(X_train,y_train)


#print(ridge_model.score(X_train,y_train))
#print(lasso_model.score(X_train,y_train))
print(elas_model.score(X_train,y_train))

#print(ridge_model.score(X_test,y_test))
#print(lasso_model.score(X_test,y_test))
print(elas_model.score(X_test,y_test))

    #SONUCLAR#
#0.7598356729443297
#0.6641414096026015
#0.6659367150921337
#0.75701870324654
#0.6893625330713165
#0.6891786972018532

#tahminrid=ridge_model.predict(X_test)
#tahminlasso=lasso_model.predict(X_test)
tahminelas=elas_model.predict(X_test)

#print(mt.mean_squared_error(y_test,tahminrid))
#print(mt.mean_squared_error(y_test,tahminlasso))
print(mt.mean_squared_error(y_test,tahminelas))

   #SONUCLAR#
#0.057080786317968105
#0.07297446803118428
#0.07301765446620055

lamb=ElasticNetCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_

elas_model2=ElasticNet(alpha=lamb)
elas_model2.fit(X_train,y_train)

print(elas_model2.score(X_train,y_train))
print(elas_model2.score(X_test,y_test))

tahminelas2=elas_model2.predict(X_test)
print(mt.mean_squared_error(y_test,tahminelas2))