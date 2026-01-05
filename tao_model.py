import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Đọc dữ liệu với dấu phân cách là ';'
print("📥 Đang đọc dữ liệu secondary_data.csv...")
dataset = pd.read_csv("secondary_data.csv", sep=';')

# 2. Xử lý dữ liệu thiếu (NaN)
# Dữ liệu nấm thường có nhiều ô trống, ta sẽ điền bằng 'unknown' hoặc giá trị phổ biến
dataset = dataset.fillna('unknown')

# 3. Mã hóa dữ liệu (Label Encoding)
# Vì hầu hết các cột là dạng chữ (categorical), ta cần chuyển sang số
print("🧪 Đang mã hóa dữ liệu...")
encoders = {}
for column in dataset.columns:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column].astype(str))
    encoders[column] = le

# Xác định mục tiêu (y là cột 'class') và đặc trưng (X là các cột còn lại)
X = dataset.drop('class', axis=1).values
y = dataset['class'].values

# Lưu lại định nghĩa của lớp (ví dụ: 0 là edible, 1 là poisonous)
target_le = encoders['class']
print(f"📘 Định nghĩa lớp: {list(target_le.classes_)}")

# 4. Chia dữ liệu Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# 5. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Huấn luyện mô hình Random Forest
print("🏗️ Đang huấn luyện mô hình...")
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
classifier.fit(X_train, y_train)

# 7. Lưu mô hình và scaler
joblib.dump(classifier, "mushroom_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"✅ Đã lưu mô hình! Độ chính xác trên tập test: {classifier.score(X_test, y_test):.2%}")