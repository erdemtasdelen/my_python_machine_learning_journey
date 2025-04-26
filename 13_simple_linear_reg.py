import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


#VERİ OKU
data=pd.read_csv("/Users/erdemtasdelen/Downloads/maas.csv")
veri=data.copy()


#DEPENDENT AND INDEPENDENT VARIABLES
y=veri["Salary"]
X=veri[["YearsExperience"]]


#GRAFİK ÇİZ
#plt.scatter(X,y)
#plt.show()


#OLS MODEL
sabit=sm.add_constant(X)
model=sm.OLS(y,sabit).fit()
print(model.summary())

#Scikit-learn modeli
lr=LinearRegression()

print(X.values.reshape(-1,1))