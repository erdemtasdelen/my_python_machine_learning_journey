import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt


data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/position_salaries.csv")
veri=data.copy()
print(veri)

y=veri["Salary"]
X=veri["Level"]


y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)

scx=StandardScaler()
scy=StandardScaler()

y=scy.fit_transform(y)
X=scx.fit_transform(X)

svrmodel=SVR(kernel="rbf")
svrmodel.fit(X,y)
tahmin=svrmodel.predict(X)

plt.scatter(X,y,color="red")
plt.plot(X,tahmin)
plt.show()


