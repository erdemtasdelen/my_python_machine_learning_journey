import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as mt
from sklearn.ensemble import BaggingRegressor
import numpy as np

data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/Advertising.csv")
veri=data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)

r2=mt.r2_score(y_test,dttahmin)
rmse=np.sqrt(mt.mean_squared_error(y_test, dttahmin))



bgmodel=BaggingRegressor(estimator=DecisionTreeRegressor(),random_state=0)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)

r2_bg=mt.r2_score(y_test,bgtahmin)
rmse_bg=np.sqrt(mt.mean_squared_error(y_test,bgtahmin))

print("Bagging R2: {}  Bagging RMSE: {}".format(r2_bg,rmse_bg))


parametreler1={"estimator__min_samples_split":range(2,25)}
grid1=GridSearchCV(estimator=bgmodel,param_grid=parametreler1,cv=10)
grid1.fit(X_train,y_train)
print(grid1.best_params_)


parametreler2={"n_estimators":range(2,25)}
grid2=GridSearchCV(estimator=bgmodel,param_grid=parametreler2,cv=10)
grid2.fit(X_train,y_train)
print(grid2.best_params_)





