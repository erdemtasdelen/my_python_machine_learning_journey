            #ÖZET#
#bu kod,etiketli bir veri kümesi üzer,nde LDA(linear discriminant analysıs)uygulayarak boyut indirgeme gerçeklestiriyor.
#PCAdan farklı olarak LDA hem ozellık matrisini(X) hem de sınıf etiketlerini(y) kullanır ve sınıflar arasındaki farkı maksimize eden dogrusal bileşenler üretir
#verilen standardize edildikten sonra LDA ile daha az sayıda ama sınfılar acısından daha anamlı oznıtelıkler olusturulmus olur.
##bu islem sonucunde modelleme daha hızlı, daha sade ve sınuflandırma acısından daha etkılı hale gelir




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/red_wine_quality.csv")
veri=data.copy()

y=veri["quality"]
X=veri.drop(columns="quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#ÖLÇEKLEME(STANDARTİZASYON)
#LDA ve PCA gibi boyut indirgeme algoritmaları, ölçek farklarına duyarlıdır
#o yuzden tum featureler aynı ölçekte olmalı.. StandartScaler ile normalize ettik
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#LDA MODELİ TANIMLAMA#
#LDA sınıflar arası ayrımı maksimize etmeye calısır
#LDA maks bileşen sayısı n_classes-1 olabilir!
lda=LinearDiscriminantAnalysis(n_components=4)



#PCA'dan farkı burada baslıyor:
#PCA sadece X'e bakar, ama LDA X+Yyi birlikte alır cünkü 'sınıf ayrımını optimize eder.'
X_train2=lda.fit_transform(X_train,y_train)   #--> LDA yapısı kullanıyorsak etıketlenmis yapı(y)yi de burada almamız gerekiyor. PCAda tek X yetiyodı
X_test2=lda.transform(X_test)


print(np.cumsum(lda.explained_variance_ratio_)*100)