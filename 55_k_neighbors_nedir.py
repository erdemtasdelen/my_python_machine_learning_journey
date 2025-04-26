import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/cancer_data.csv")
veri=data.copy()

M=veri[veri["diagnosis"]=="M"]
B=veri[veri["diagnosis"]=="B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Köyü Huylu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="İyi Huylu")
plt.legend()
plt.show()

#--> k = komşuluk sayısı

