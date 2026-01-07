from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- 1. CẤU HÌNH LOGGING ---
# Giúp theo dõi các hoạt động và lỗi của ứng dụng trong terminal/console
logging.basicConfig(level=logging.INFO)

# --- 2. TẢI MÔ HÌNH VÀ CÁC THÀNH PHẦN ---
# Xác định thư mục hiện tại để tìm file mô hình .pkl chính xác
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mushroom_random_forest_model.pkl")

model_bundle = None
# Kiểm tra nếu file mô hình tồn tại thì mới nạp vào bộ nhớ
if os.path.exists(MODEL_PATH):
    try:
        # Nạp "chiếc hộp" pkl chứa: Model AI, Bộ chuẩn hóa (Scaler), Bộ mã hóa (Encoders)
        model_bundle = joblib.load(MODEL_PATH)
        logging.info(f"✅ Đã nạp thành công mô hình: {type(model_bundle['classifier']).__name__}")
    except Exception as e:
        logging.error(f"❌ Lỗi khi nạp file model: {e}")
else:
    logging.error(f"❌ Không tìm thấy file {MODEL_PATH}. Hãy chạy train.py trước!")

# --- 3. CÁC ĐỊNH TUYẾN (ROUTES) ---

@app.route("/")
def index():
    """Route này trả về giao diện trang chủ (file index.html)"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Xử lý dữ liệu gửi từ Form và trả về kết quả dự đoán dạng JSON"""
    if not model_bundle:
        return jsonify({"success": False, "error": "Hệ thống AI chưa sẵn sàng. Kiểm tra file .pkl!"})

    try:
        # Lấy các thành phần đã đóng gói từ file pkl
        model = model_bundle["classifier"]      # Mô hình Random Forest
        scaler = model_bundle["scaler"]          # Bộ thước đo chuẩn hóa số
        encoders = model_bundle["encoders"]      # Bộ dịch mã chữ (cap-shape, color...)
        feature_names = model_bundle["features"] # Danh sách tên 6 cột đặc trưng
        target_mapping = model_bundle["target_mapping"] # { 'e': 0, 'p': 1 }
        
        # Đảo ngược bảng đối chiếu để chuyển số (0,1) về lại chữ (e,p)
        inverse_target_mapping = {v: k for k, v in target_mapping.items()}

        # 1. Thu thập dữ liệu từ Form (Web) gửi lên dưới dạng Dictionary
        form_data = request.form.to_dict()
        input_row = []
        
        # 2. Vòng lặp tiền xử lý dữ liệu theo đúng thứ tự Feature mà AI yêu cầu
        for col in feature_names:
            val = form_data.get(col, '').strip()
            
            if col in encoders:
                # Nếu là cột dạng chữ (cap-shape, color, season):
                le = encoders[col]
                char_val = val if val != '' else 'unknown'
                
                # Nếu người dùng chọn giá trị lạ, ép về nhãn 'unknown' để tránh lỗi sập web
                if char_val not in le.classes_:
                    char_val = 'unknown'
                
                # Chuyển chữ thành số bằng bộ mã hóa đã lưu
                input_row.append(le.transform([char_val])[0])
            else:
                # Nếu là cột dạng số (đường kính, chiều cao...):
                try:
                    # Chuyển chuỗi nhập vào thành số thực (float)
                    input_row.append(float(val) if val != '' else 0.0)
                except:
                    # Nếu nhập sai định dạng, mặc định trả về 0.0
                    input_row.append(0.0)

        # 3. CHUYỂN ĐỔI SANG DATAFRAME: Gắn lại tên cột cho dữ liệu để Scaler hoạt động chính xác
        # Bước này cực kỳ quan trọng để tránh lỗi "Feature names mismatch"
        input_df = pd.DataFrame([input_row], columns=feature_names)

        # 4. CHUẨN HÓA DỮ LIỆU: Dùng bộ Scaler (thước đo) đã học được từ lúc Train
        input_scaled = scaler.transform(input_df)

        # 5. DỰ ĐOÁN: AI đưa ra kết quả cuối cùng (trả về số 0 hoặc 1)
        prediction_num = model.predict(input_scaled)[0]
        
        # 6. TÍNH ĐỘ TIN CẬY: Lấy xác suất cao nhất trong các cây quyết định
        try:
            probs = model.predict_proba(input_scaled)[0]
            confidence = float(np.max(probs)) * 100
        except:
            confidence = 100.0

        # 7. GIẢI MÃ KẾT QUẢ: Chuyển từ số (0,1) về thông điệp dễ hiểu cho người dùng
        result_label = inverse_target_mapping.get(prediction_num)
        is_edible = (result_label == 'e')
        
        result_text = "ĂN ĐƯỢC ✅" if is_edible else "CÓ ĐỘC - NGUY HIỂM 💀"
        class_css = "text-success" if is_edible else "text-danger"

        # In log ra terminal để bạn dễ dàng theo dõi dữ liệu đang xử lý
        logging.info(f"Dữ liệu nhập: {input_row} => Kết quả: {result_text} ({confidence:.1f}%)")

        # 8. TRẢ KẾT QUẢ: Gửi dữ liệu về lại cho JavaScript trên trình duyệt hiển thị
        return jsonify({
            "success": True,
            "variety": result_text,
            "class_css": class_css,
            "confidence": f"{confidence:.1f}%"
        })

    except Exception as e:
        # Nếu có bất kỳ lỗi nào xảy ra trong quá trình trên, ghi lại log và báo lỗi
        logging.error(f"Lỗi xử lý dự đoán: {e}")
        return jsonify({"success": False, "error": f"Lỗi hệ thống: {str(e)}"})

# --- 4. CHẠY SERVER ---
if __name__ == "__main__":
    # Lấy cổng (Port) từ hệ thống (Dùng cho Render) hoặc mặc định là 5000 (Local)
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' để server có thể truy cập được từ bên ngoài internet
    # debug=True để tự động tải lại code khi bạn lưu file và hiện lỗi chi tiết
    app.run(host='0.0.0.0', port=port, debug=True)