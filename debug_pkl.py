import joblib
import pandas as pd
import numpy as np

def verify_pkl_with_csv(pkl_path, csv_path):
    print(f"--- BAT DAU KIEM TRA: {pkl_path} ---")
    try:
        # 1. Tai file bundle .pkl
        data = joblib.load(pkl_path)
        model = data["classifier"]
        scaler = data["scaler"]
        encoders = data["encoders"]
        feature_names = data["features"]
        mapping = data["target_mapping"]

        # 2. Doc file CSV de lay mau thuc te
        df = pd.read_csv(csv_path, sep=';')
        
        # Lay 5 mau an duoc (e) va 5 mau doc (p)
        samples_e = df[df['class'] == 'e'].head(5)
        samples_p = df[df['class'] == 'p'].head(5)
        test_df = pd.concat([samples_e, samples_p])

        print(f"✅ Da tai 10 mau tu CSV. Dang tien hanh doi chieu...\n")
        
        results = []

        # 3. Lap qua tung mau de kiem tra
        for _, row in test_df.iterrows():
            actual_class = row['class']
            
            # Xu ly hang du lieu theo dung thu tu feature AI da hoc
            input_row = []
            for col in feature_names:
                val = row[col]
                if col in ['cap-diameter', 'stem-height', 'stem-width']:
                    input_row.append(float(val) if pd.notnull(val) else 0.0)
                else:
                    le = encoders[col]
                    char_val = str(val) if pd.notnull(val) else 'unknown'
                    # Dung le.transform de chuyn chu thanh so
                    input_row.append(le.transform([char_val])[0])

            # 4. Chuan hoa (Scaling) va Du doan
            input_scaled = scaler.transform([input_row])
            prediction_num = model.predict(input_scaled)[0]
            
            # Dich ket qua tu so sang chu 'e' hoac 'p'
            predicted_class = 'e' if prediction_num == mapping['e'] else 'p'
            
            # Ghi lai thong so de in bang
            results.append({
                "Thuc te": "AN DUOC" if actual_class == 'e' else "CO DOC",
                "AI Doan": "AN DUOC" if predicted_class == 'e' else "CO DOC",
                "Ket qua": "✅ KHOP" if actual_class == predicted_class else "❌ SAI",
                "Diam": row['cap-diameter'],
                "Shape": row['cap-shape'],
                "Color": row['cap-color'],
                "Season": row['season']
            })

        # 5. In bang ket qua
        final_df = pd.DataFrame(results)
        print(final_df.to_string(index=False))
        
        print("\n" + "="*50)
        print(f"📘 Thong tin Mapping trong PKL: {mapping}")
        print("="*50)

    except Exception as e:
        print(f"❌ Loi trong qua trinh kiem tra: {e}")

if __name__ == "__main__":
    verify_pkl_with_csv("mushroom_final_model.pkl", "secondary_data.csv")