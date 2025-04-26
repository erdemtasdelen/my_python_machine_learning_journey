#Bu analizde, seaborn kütüphanesine gömülü tips veri setini kullanarak müşterilerin bıraktıgı tip miktarını tahmın etmeyi amaçladık.
#ilk olarak veriyi pandas ile okuduk ve orijinal halini koruyarak üzerinde calısacılecegımzı bir kopyasını olusturduk
#ardındani modelimizin sayısal verilerle calısabilmesi icin kategorik degiskenleri get_dummıes() fonksıyonuyla 0-1 formatına donusturduk
#bagımlı degısken olarak tip sutünunu seçip geri kalan sutğnlar bagımsız degısken (features ) olarak ayırdık.
#veriyi %80train %20test olarak ikiye böldük ve linear regression modeliyle egittik.
#modelin test verisi uzerındeki tahmınleri ile gercek degerleri karsılastırarak bir cizgi grafi cizdik ve performansı r2_score ile degerlendirdik
#elde edilen r2 skoru, modelin tip degiskenini acıklama gücünü sayısal olarak gostermektedir



import pandas as pd 
import matplotlib.pyplot as plt  #-->grafik cizmek icin temel kütüphane
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt    #-->modelin başarısını ölçmek icin kullanılır.

data=sns.load_dataset("tips")  #--> sns.load_datasets(), hazir veri seti çeker.
veri=data.copy()

#fit(), egitir :::: predict(), tahmin eder


#print(veri.isnull().sum())
#print(veri.dtypes)

kategori=[]
kategorik=veri.select_dtypes(include=["category"])

for i in kategorik.columns:
    kategori.append(i)

#print(kategori)




#get_dummies(), kategorık sutünları binary(0-1) sutünlara çevirir.

veri=pd.get_dummies(veri,columns=kategori,drop_first=True)


y=veri["tip"]
X=veri.drop(columns="tip",axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)  #-->model X_traindeki verilerle y_traini eşleştirerek öğrenir
tahmin=lr.predict(X_test)

y_test=y_test.sort_index()  #-->tahmin edilen degerlerin sıralamasını bozmayalım diye y_testi indexe gçre sıralıyoruz
df=pd.DataFrame({"Gerçek":y_test,"Tahmin":tahmin})  #-->gerçek ve tahmın degerlerini aynı dataframee koyarak grafik çizecegiz
df.plot(kind="line")

print(mt.r2_score(y_test,tahmin))  #-->r2 skoru ne kadar 1'e yakınsa model o kadar iyi demektir. örnegin r2=0.75 ise %75 başarıyla tahmin ediyoruz demektir.