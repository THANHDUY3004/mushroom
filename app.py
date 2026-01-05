from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import logging
import joblib

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 1. Load Mô hình và Bộ chuẩn hóa (Scaler)
try:
    model = joblib.load("mushroom_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    # Tải lại bộ mã hóa nhãn (LabelEncoders) nếu bạn lưu chúng, 
    # hoặc dùng một hàm map đơn giản cho các giá trị chữ.
    logging.info("✅ Mô hình và Scaler đã được tải thành công.")
except Exception as e:
    logging.error("❌ Không thể tải file: %s", e)
    model = None

# Định nghĩa nhãn đầu ra dựa trên file secondary_data.csv
# Thông thường: p = poisonous (độc), e = edible (ăn được)
class_map = {0: "Ăn được (Edible)", 1: "Có độc (Poisonous)"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Mô info mô hình chưa sẵn sàng."}), 500

    try:
        # Lấy toàn bộ dữ liệu từ form (giả sử form gửi đủ các trường)
        data = request.form.to_dict()
        
        # Chuyển đổi dữ liệu form thành DataFrame (theo đúng thứ tự cột khi train)
        # Lưu ý: Bạn cần xử lý mã hóa các ký tự chữ (x, f, s...) thành số tương ứng
        # Ở đây là ví dụ minh họa cách nhận 3 đặc trưng quan trọng nhất:
        input_values = [
            float(data.get('cap_diameter', 0)),
            # Với các cột dạng chữ, bạn cần map từ ký tự sang số đã dùng khi train
            # Ví dụ: 'x' -> 5, 's' -> 2 (phụ thuộc vào LabelEncoder của bạn)
            ord(data.get('cap_shape', 'x')[0]), 
            float(data.get('stem_height', 0)),
            float(data.get('stem_width', 0)),
            # ... Thêm đủ 20 cột theo đúng thứ tự của file CSV
        ]

        # Giả sử bạn đã chuẩn bị mảng đủ 20 đặc trưng:
        # features = np.array([input_values])
        
        # Đơn giản nhất cho việc demo: Lấy các giá trị số và điền 0 cho các cột thiếu
        full_features = np.zeros(20) 
        full_features[0] = float(data.get('cap-diameter', 0))
        full_features[8] = float(data.get('stem-height', 0))
        full_features[9] = float(data.get('stem-width', 0))
        
        # Chuẩn hóa
        features_scaled = scaler.transform([full_features])

        # Dự đoán
        prediction = model.predict(features_scaled)[0]
        result = class_map.get(int(prediction), "Không xác định")

        return jsonify({"success": True, "variety": result})

    except Exception as e:
        logging.error("Lỗi dự đoán: %s", e)
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)