from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging

app = Flask(__name__)

# Cau hinh logging de theo doi tren Terminal
logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. TẢI MÔ HÌNH VÀ CÁC THÀNH PHẦN
# ==========================================
MODEL_PATH = "mushroom_final_model.pkl"

try:
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["classifier"]
        scaler = bundle["scaler"]
        encoders = bundle["encoders"]
        feature_names = bundle["features"]
        target_mapping = bundle["target_mapping"]
        logging.info(f"✅ Da tai model. Mapping: {target_mapping}")
    else:
        logging.error("❌ Khong tim thay file mushroom_final_model.pkl")
        model = None
except Exception as e:
    logging.error(f"❌ Loi load model: {e}")
    model = None

# ==========================================
# 2. HÀM GỢI Ý TÊN NẤM (DỰA TRÊN LOGIC)
# ==========================================
def get_mushroom_name(diam, shape, color, season):
    # Logic nay hoat dong doc lap voi AI de goi y ten goi
    if shape == 'x' and color == 'n' and season in ['a', 'w']:
        return "Nam Huong (Shiitake)"
    elif shape == 'b' and season == 'u':
        return "Nam Rom (Paddy Straw)"
    elif shape == 'x' and color == 'w' and diam < 8:
        return "Nam Mo (White Button)"
    elif shape == 'f' and diam > 10:
        return "Nam Dui Ga / Bao Ngu"
    elif color == 'r' and shape == 'x':
        return "Nam Tan Doc (Amanita - Nguy hiem!)"
    return "Chua xac dinh ten loai"

# ==========================================
# 3. ROUTES
# ==========================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model chua duoc tai!"})

    try:
        # Lay du lieu tu Form
        form_data = request.form.to_dict()
        
        # CHUAN BI MANG 20 COT (Phai dung thu tu luc Train)
        input_row = []
        
        for col in feature_names:
            val = form_data.get(col)
            
            # Neu la cot so (Diameter, Height, Width)
            if col in ['cap-diameter', 'stem-height', 'stem-width']:
                try:
                    input_row.append(float(val) if val and val.strip() != '' else 0.0)
                except:
                    input_row.append(0.0)
            
            # Neu la cot chu (Categorical)
            else:
                le = encoders.get(col)
                # Neu o trong, dung 'unknown'
                char_val = str(val) if (val and val.strip() != '') else 'unknown'
                
                try:
                    # Chuyen chu thanh so bang Encoder da hoc
                    encoded_val = le.transform([char_val])[0]
                    input_row.append(encoded_val)
                except:
                    # Phong truong hop nhan la, dung nhan cua 'unknown'
                    unknown_val = le.transform(['unknown'])[0]
                    input_row.append(unknown_val)

        # CHUAN HOA VA DU DOAN
        input_scaled = scaler.transform([input_row])
        prediction_num = model.predict(input_scaled)[0]
        
        # DICH KET QUA (Dua tren target_mapping {'e': 0, 'p': 1})
        # prediction_num == 0 nghia la 'e' (Edible)
        if prediction_num == target_mapping.get('e', 0):
            result_text = "An duoc (Edible) ✅"
        else:
            result_text = "Co doc (Poisonous) 💀"

        # GOI Y TEN LOAI
        suggested = get_mushroom_name(
            float(form_data.get('cap-diameter', 0)),
            form_data.get('cap-shape', 'x'),
            form_data.get('cap-color', 'n'),
            form_data.get('season', 'a')
        )

        return jsonify({
            "success": True,
            "variety": result_text,
            "suggested_name": suggested
        })

    except Exception as e:
        logging.error(f"Loi predict: {e}")
        return jsonify({"success": False, "error": str(e)})
def quick_test():
    # 1. Tai file model
    try:
        data = joblib.load("mushroom_final_model.pkl")
        model = data["classifier"]
        scaler = data["scaler"]
        encoders = data["encoders"]
        feature_names = data["features"]
        mapping = data["target_mapping"]
        print("✅ Da tai model thanh cong!")
    except:
        print("❌ Khong tim thay file mushroom_final_model.pkl")
        return

    # 2. Dinh nghia cac kich ban test (Gia lap du lieu tu Form)
    test_cases = [
        {
            "name": "Kich ban 1: Nam Huong (An duoc)",
            "data": {'cap-diameter': 15.0, 'stem-height': 16.0, 'stem-width': 17.0, 
                     'cap-shape': 'x', 'cap-color': 'n', 'season': 'a'}
        },
        {
            "name": "Kich ban 2: Nam Doc (Mau doc)",
            "data": {'cap-diameter': 2.5, 'stem-height': 4.8, 'stem-width': 3.2, 
                     'cap-shape': 'b', 'cap-color': 'n', 'season': 'u'}
        }
    ]

    print("\n--- BAT DAU KIEM TRA NHANH ---")
    
    for case in test_cases:
        print(f"\n[Testing: {case['name']}]")
        
        # Chuan bi mang 20 phan tu dung thu tu features
        input_row = []
        for col in feature_names:
            val = case['data'].get(col)
            
            if col in ['cap-diameter', 'stem-height', 'stem-width']:
                input_row.append(float(val) if val else 0.0)
            else:
                le = encoders.get(col)
                # Neu thong so khong co trong kich ban, dung 'unknown'
                char_val = str(val) if val else 'unknown'
                input_row.append(le.transform([char_val])[0])
        
        # Chuan hoa va du doan
        input_scaled = scaler.transform([input_row])
        pred = model.predict(input_scaled)[0]
        
        # Dich ket qua
        result = "AN DUOC ✅" if pred == mapping['e'] else "CO DOC 💀"
        print(f"-> Ket qua AI tra ve: {result}")
if __name__ == "__main__":
    quick_test()
    print("\n🚀 Mushroom AI dang chay tai: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)