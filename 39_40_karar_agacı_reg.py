import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt


data=pd.read_csv("/Users/erdemtasdelen/Downloads/maas_sample.csv")
veri=data.copy()

y=veri["Egitim_Yili"]
X=veri["Maas"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)

dtr=DecisionTreeRegressor(random_state=0,max_leaf_nodes=3)
dtr.fit(X,y)
tahmin=dtr.predict(X)

plt.figure(figsize=(20,10),dpi=100)
plot_tree(dtr,feature_names=["Maas"],class_names=["Egitim_Yili"],rounded=True,filled=True)
plt.show()

