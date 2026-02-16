import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv(r"C:\Users\muham\Downloads\archive (9)\data.csv")

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("Doğruluk Oranı:", accuracy_score(y_test, knn.predict(X_test)))


yeni_hasta = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
                        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])

yeni_hasta_df = pd.DataFrame(yeni_hasta, columns=X.columns)

yeni_hasta_scaled = scaler.transform(yeni_hasta_df)

sonuc = knn.predict(yeni_hasta_scaled)

print("Tahmin:", "KÖTÜ HUYLU (M)" if sonuc[0] == 1 else "İYİ HUYLU (B)")










