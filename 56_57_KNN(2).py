import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data=pd.read_csv("/Users/erdemtasdelen/Desktop/python_lessons/data/cancer_data.csv")
veri=data.copy()


veri=veri.drop(columns=["id","Unnamed: 32"],axis=1)


#    diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  ...  compactness_worst  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_worst
#0           M        17.99         10.38          122.80     1001.0  ...            0.66560           0.7119                0.2654          0.4601                  0.11890
#1           M        20.57         17.77          132.90     1326.0  ...            0.18660           0.2416                0.1860          0.2750                  0.08902
#2           M        19.69         21.25          130.00     1203.0  ...            0.42450           0.4504                0.2430          0.3613                  0.08758
#3           M        11.42         20.38           77.58      386.1  ...            0.86630           0.6869                0.2575          0.6638                  0.17300
#4           M        20.29         14.34          135.10     1297.0  ...            0.20500           0.4000                0.1625          0.2364                  0.07678
#..        ...          ...           ...             ...        ...  ...                ...              ...                   ...             ...                      ...
#564         M        21.56         22.39          142.00     1479.0  ...            0.21130           0.4107                0.2216          0.2060                  0.07115
#565         M        20.13         28.25          131.20     1261.0  ...            0.19220           0.3215                0.1628          0.2572                  0.06637
#566         M        16.60         28.08          108.30      858.1  ...            0.30940           0.3403                0.1418          0.2218                  0.07820
#567         M        20.60         29.33          140.10     1265.0  ...            0.86810           0.9387                0.2650          0.4087                  0.12400
#568         B         7.76         24.54           47.92      181.0  ...            0.06444           0.0000                0.0000          0.2871                  0.07039


veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]


y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=KNeighborsClassifier(n_neighbors=9)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)

acs=accuracy_score(y_test,tahmin)
print(acs*100)   #-->94.73684210526315

basarı=[]

for k in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    tahmin2=knn.predict(X_test)
    basarı.append(accuracy_score(y_test,tahmin2))

#print(max(basarı))  #-->0.964912280701754

plt.plot(range(1,20),basarı)
plt.xlabel("K")
plt.ylabel("Başarı")
plt.show()
