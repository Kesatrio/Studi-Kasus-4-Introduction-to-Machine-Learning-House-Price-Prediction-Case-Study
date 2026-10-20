import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
raw_data = {
    'Luas_Rumah': [1500, 1800, 2400, 3000, 3500, 1200, 2000, 2800, 1600, 2200],
    'Jumlah_Kamar': [3, 3, 4, 4, 5, 2, 3, 4, 3, 3],
    'Jarak_Pusat_Kota': [10.5, 8.0, 5.2, 3.0, 2.0, 12.0, 7.5, 4.0, 9.0, 6.0],
    'Harga_Rumah': [220000, 260000, 350000, 450000, 520000, 180000, 290000, 410000, 240000, 310000]
}
df = pd.DataFrame(raw_data)
df = df.dropna().drop_duplicates()
X = df[['Luas_Rumah', 'Jumlah_Kamar', 'Jarak_Pusat_Kota']]
y = df['Harga_Rumah']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
rumah_baru = pd.DataFrame({'Luas_Rumah': [2100], 'Jumlah_Kamar': [3], 'Jarak_Pusat_Kota': [5.0]})
prediksi = model.predict(rumah_baru)[0]
print(f"Hasil Prediksi Harga Rumah Baru: ${prediksi:,.2f}")