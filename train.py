import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==========================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
print("📥 Đang tải dữ liệu...")
# Đảm bảo file csv nằm cùng thư mục
df = pd.read_csv("secondary_data.csv", sep=';')

# Tách nhãn và đặc trưng
X = df.drop('class', axis=1)
y = df['class']

# Mã hóa nhãn mục tiêu (e=0, p=1)
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print(f"📘 Bản đồ nhãn mục tiêu: {target_mapping}")

# Phân loại cột số và cột chữ
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

# Xử lý giá trị thiếu ban đầu
X[cat_cols] = X[cat_cols].fillna('unknown')
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# ==========================================
# 2. MÃ HÓA ĐẶC TRƯNG (TRÁNH LỖI UNKNOWN)
# ==========================================
print("⚙️ Đang mã hóa đặc trưng...")
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # QUAN TRỌNG: Ép LabelEncoder học chữ 'unknown' ngay từ đầu
    unique_values = X[col].astype(str).unique()
    if 'unknown' not in unique_values:
        unique_values = np.append(unique_values, 'unknown')
    
    le.fit(unique_values)
    X[col] = le.transform(X[col].astype(str))
    encoders[col] = le

# ==========================================
# 3. CHIA DỮ LIỆU VÀ CHUẨN HÓA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH
# ==========================================
print("🏗️ Đang huấn luyện Random Forest (Vui lòng đợi)...")
# Sử dụng class_weight='balanced' để tránh thiên kiến nấm ăn/độc
model = RandomForestClassifier(
    n_estimators=200, 
    criterion='entropy',
    class_weight='balanced', 
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ==========================================
# 5. LƯU BUNDLE MODEL
# ==========================================
model_bundle = {
    "classifier": model,
    "scaler": scaler,
    "encoders": encoders,
    "features": list(X.columns),
    "target_mapping": target_mapping
}

joblib.dump(model_bundle, "mushroom_final_model.pkl")
print(f"✅ Đã lưu file: mushroom_final_model.pkl")
print(f"📊 Độ chính xác: {model.score(X_test_scaled, y_test):.2%}")

# ==========================================
# 6. IN THÔNG SỐ ĐỂ BẠN NHẬP WEB TEST
# ==========================================
def print_test_samples():
    print("\n" + "="*85)
    print("🔍 DANH SÁCH MẪU ĐỂ TEST WEB (NHẬP CHÍNH XÁC CÁC SỐ NÀY)")
    print("-" * 85)
    print(f"{'STT':<4} | {'LOẠI':<10} | {'DIAM':<6} | {'HEIGHT':<6} | {'WIDTH':<6} | {'SHAPE':<5} | {'COLOR':<5} | {'SEASON'}")
    print("-" * 85)
    
    samples_e = df[df['class'] == 'e'].head(5)
    samples_p = df[df['class'] == 'p'].head(5)
    test_set = pd.concat([samples_e, samples_p])
    
    for i, (_, row) in enumerate(test_set.iterrows(), 1):
        loai = "ĂN ĐƯỢC" if row['class'] == 'e' else "CÓ ĐỘC"
        print(f"{i:<4} | {loai:<10} | {row['cap-diameter']:<6} | {row['stem-height']:<6} | {row['stem-width']:<6} | {row['cap-shape']:<5} | {row['cap-color']:<5} | {row['season']}")
    
    print("="*85)
    print("💡 Lưu ý: Nếu nhập đúng STT 1-5 mà Web vẫn báo CÓ ĐỘC, hãy kiểm tra lại class_map trong app.py")

if __name__ == "__main__":
    print_test_samples()