import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt


data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/red_wine_quality.csv")
veri=data.copy()

#kor=veri.corr()
#sns.heatmap(kor,annot=True,cbar=True)
#plt.show()


y=veri["quality"]
X=veri.drop(columns="quality",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

pca=PCA(n_components=11)  #->(1279, 2) degisken sayısını 2ye indirdik. bu bizim belirledigimiz bir hiper parametre

X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

#print(X_train.shape)   #DataSetimiz: (1279, 11) yani 1279 gozlem, 11 degiskenden olusuyor.
#print(X_train2.shape)



print(np.cumsum(pca.explained_variance_ratio_)*100)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel("Bileşen Sayısı")
#plt.ylabel("Açıklanan Varyans")
#plt.show()

        
#[ 28.01769042  45.58168574  59.53932155  70.62114399  79.6423922
#  85.55109031  90.81771725  94.70160536  97.83107305  99.43207028
# 100.        ]
#--> burada şu anlatılıyor: eger 11 degiskenin hepsini kullanırsan model 100, 10unu kullanırsan 99, 1ini kullanırsan 28 oranla acıklayabilir model

lm=LinearRegression()
lm.fit(X_train2,y_train)
tahmin=lm.predict(X_test2)

r2=mt.r2_score(y_test,tahmin)
mse = mt.mean_squared_error(y_test, tahmin)
rmse = np.sqrt(mse)

print("R2: {}   RMSE: {}".format(r2,rmse))

cv=KFold(n_splits=10,shuffle=True,random_state=1)

lm2=LinearRegression()
RMSE=[]

for i in range(1,X_train2.shape[1]+1):
    hata=np.sqrt(-1*cross_val_score(lm2,X_train2[:,:i],y_train.ravel(),
    cv=cv,scoring="neg_mean_squared_error").mean())
    RMSE.append(hata)


plt.plot(RMSE,"-x")
plt.xlabel("Bileşen Sayısı")
plt.ylabel("RMSE")
plt.show()