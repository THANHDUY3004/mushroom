import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    # --- BƯỚC 1: ĐỌC DỮ LIỆU THÔ ---
    print("🚀 Đang đọc dữ liệu từ secondary_data.csv...")
    try:
        # Sử dụng sep=';' vì file CSV của bạn dùng dấu chấm phẩy để ngăn cách các cột
        df = pd.read_csv('secondary_data.csv', sep=';')
    except Exception as e:
        print(f"❌ Lỗi: Không tìm thấy hoặc không thể đọc file CSV. {e}")
        return

    # --- BƯỚC 2: CHỌN LỌC ĐẶC TRƯNG (FEATURE SELECTION) ---
    # Chúng ta chỉ chọn 6 thông số quan trọng nhất để người dùng nhập trên Web dễ dàng
    features = ['cap-diameter', 'cap-shape', 'cap-color', 'stem-height', 'stem-width', 'season']
    target = 'class' # Cột mục tiêu: 'p' (độc) hoặc 'e' (ăn được)

    # Loại bỏ những dòng nấm bị thiếu thông tin (NaN) ở các cột đã chọn để dữ liệu "sạch" hơn
    df = df.dropna(subset=features + [target])

    X = df[features].copy() # Dữ liệu đầu vào (6 thông số)
    y = df[target].copy()   # Kết quả thực tế (nhãn)

    # --- BƯỚC 3: MÃ HÓA DỮ LIỆU CHỮ (ENCODING) ---
    # AI không hiểu chữ 'x', 'f', 'n'... nên ta phải chuyển chúng thành số 0, 1, 2...
    encoders = {}
    categorical_cols = ['cap-shape', 'cap-color', 'season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Thêm nhãn 'unknown' dự phòng cho trường hợp người dùng nhập giá trị lạ trên Web
        unique_values = list(X[col].unique()) + ['unknown']
        le.fit(unique_values)
        X[col] = le.transform(X[col])
        encoders[col] = le # Lưu lại bộ giải mã để app.py dùng sau này

    # Chuyển nhãn 'e', 'p' thành số 0 và 1
    target_le = LabelEncoder()
    y = target_le.fit_transform(y)
    # Lưu lại bảng đối chiếu: ví dụ {'e': 0, 'p': 1}
    target_mapping = dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))

    # --- BƯỚC 4: CHIA DỮ LIỆU TRAIN/TEST ---
    # Chia 80% dữ liệu để AI học, 20% dữ liệu để chấm điểm năng lực của AI
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- BƯỚC 5: CHUẨN HÓA SỐ LIỆU (SCALING) ---
    # Đưa đường kính (cm) và chiều cao (cm) về cùng một hệ quy chiếu (thang đo chuẩn)
    # giúp mô hình không bị thiên vị cột có con số lớn hơn.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Học và chuyển đổi tập train
    X_test = scaler.transform(X_test)       # Chỉ chuyển đổi tập test theo thước đo tập train

    # --- BƯỚC 6: XÂY DỰNG RỪNG NGẪU NHIÊN (RANDOM FOREST) ---
    print("🌲 Đang huấn luyện Rừng ngẫu nhiên (100 cây quyết định)...")
    model = RandomForestClassifier(
        n_estimators=100,      # Xây dựng 100 cây để cùng bỏ phiếu bầu kết quả
        max_depth=12,          # Giới hạn chiều cao của cây để tránh "học vẹt" dữ liệu cũ
        min_samples_split=5,   # Mỗi nhánh phải có ít nhất 5 mẫu mới được chia tiếp
        random_state=42,       # Đảm bảo kết quả giống nhau mỗi lần chạy lại code
        n_jobs=-1              # Sử dụng tối đa nhân CPU để huấn luyện nhanh nhất
    )
    # AI bắt đầu quá trình học tập tại đây
    model.fit(X_train, y_train)

    # --- BƯỚC 7: KIỂM TRA ĐỘ CHÍNH XÁC ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Huấn luyện hoàn tất!")
    print(f"📊 Độ chính xác dự đoán: {acc*100:.2f}%")
    print("\nBáo cáo chi tiết hiệu suất:")
    print(classification_report(y_test, y_pred))

    # --- BƯỚC 8: ĐÓNG GÓI THÀNH FILE .PKL ---
    # Lưu tất cả: Model, Bộ chuẩn hóa (Scaler), Bộ mã hóa (Encoders) vào 1 file duy nhất
    model_data = {
        "classifier": model,
        "scaler": scaler,
        "encoders": encoders,
        "features": features,
        "target_mapping": target_mapping
    }

    # Tên file giống với file mà app.py đang yêu cầu tải lên
    output_filename = "mushroom_random_forest_model.pkl" 
    joblib.dump(model_data, output_filename)
    print(f"💾 Đã lưu thành công: {output_filename}")

if __name__ == "__main__":
    train_model()