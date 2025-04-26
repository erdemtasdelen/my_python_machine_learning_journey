import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data=pd.read_csv("/Users/erdemtasdelen/Downloads/Advertising.csv")
veri=data.copy()
#print(veri.isnull().sum())
#print(veri.dtypes)

veri=veri.drop(["Unnamed: 0"], axis=1)
print(veri.corr()["Sales"])

#sns.pairplot(veri,kind="reg")
#plt.show()

Q1=veri["Newspaper"].quantile(0.25)
Q3=veri["Newspaper"].quantile(0.75)
IQR=Q3-Q1
ustsınır=Q3+1.5*IQR
aykırı=veri["Newspaper"]>ustsınır
veri.loc[aykırı,"Newspaper"]=ustsınır

#sns.boxplot(veri["Newspaper"])
#plt.show()

y=veri["Sales"]
X=veri[["TV","Radio"]]

#sabit=sm.add_constant(X)
#model=sm.OLS(y,sabit).fit()


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)


tahmin=lr.predict(X_test)
y_test=y_test.sort_index()



df=pd.DataFrame({"Gerçek":y_test,"Tahmin":tahmin})
df.plot(kind="line")
plt.show()