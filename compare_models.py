"""
HUẤN LUYỆN & ĐÁNH GIÁ ĐA GIẢI THUẬT
Bài toán: Dự đoán nấm ĂN ĐƯỢC hay CÓ ĐỘC

Các giải thuật sử dụng (kèm tên tiếng Việt):
1. Decision Tree        – Cây quyết định
2. Random Forest        – Rừng ngẫu nhiên
3. Logistic Regression  – Hồi quy Logistic
4. Gradient Boosting    – Tăng cường dần (Boosting)
5. Naive Bayes          – Xác suất Bayes đơn giản

Mục tiêu:
- So sánh độ tin cậy các mô hình
- Ưu tiên Recall của lớp "NẤM ĐỘC"
- Chọn mô hình an toàn nhất để triển khai
- Lưu bundle mô hình tốt nhất theo format dùng trong Flask
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import joblib

# =========================
# 1. KHAI BÁO THUỘC TÍNH
# =========================
FEATURES = [
    'cap-diameter',   # Đường kính mũ (cm)
    'cap-shape',      # Hình dạng mũ
    'cap-color',      # Màu sắc mũ
    'stem-height',    # Chiều cao thân (cm)
    'stem-width',     # Độ rộng thân (mm)
    'season'          # Mùa vụ
]
TARGET = 'class'      # Nhãn: e (ăn được), p (độc)
OUTPUT_PKL = 'mushroom_random_forest_model.pkl'  # giữ tên để tương thích với app Flask hiện có

# =========================
# 2. ĐỌC & TIỀN XỬ LÝ DỮ LIỆU
# =========================
def load_and_prepare(path='secondary_data.csv'):
    df = pd.read_csv(path, sep=';')

    # Bỏ dòng thiếu ở các cột cần thiết
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    # Tách X, y
    X = df[FEATURES].copy()
    y_raw = df[TARGET].astype(str).copy()

    # Mã hóa nhãn mục tiêu: e -> 0 (ăn được), p -> 1 (có độc)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    target_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))

    # Mã hóa thuộc tính định danh
    encoders = {}
    categorical_cols = ['cap-shape', 'cap-color', 'season']
    numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']

    for col in categorical_cols:
        le = LabelEncoder()
        # thêm 'unknown' để dự phòng input lạ khi chạy web
        unique_vals = list(X[col].astype(str).unique()) + ['unknown']
        le.fit(unique_vals)
        X.loc[:, col] = le.transform(X[col].astype(str))
        encoders[col] = le

    # Chuẩn hóa các thuộc tính số
    scaler = StandardScaler()
    X.loc[:, numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, encoders, scaler, target_mapping

# =========================
# 3. DANH SÁCH GIẢI THUẬT
# =========================
def build_models():
    return {
        "Cây quyết định (Decision Tree – Gini)": 
            DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42),

        "Cây quyết định (Decision Tree – Entropy)": 
            DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42),

        "Rừng ngẫu nhiên (Random Forest)": 
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),

        "Hồi quy Logistic (Logistic Regression)": 
            LogisticRegression(max_iter=2000, solver="liblinear"),

        "Tăng cường dần (Gradient Boosting)": 
            GradientBoostingClassifier(random_state=42),

        "Xác suất Bayes (Naive Bayes)": 
            GaussianNB()
    }


# =========================
# 4. ĐÁNH GIÁ MÔ HÌNH
# =========================
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Accuracy – Độ chính xác tổng thể
    acc = accuracy_score(y_test, y_pred)

    # Precision – Recall – F1 (macro)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )

    # Recall riêng cho lớp NẤM ĐỘC (label = 1)
    _, recall_per_class, _, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1]
    )
    recall_poisonous = float(recall_per_class[1])

    # ROC-AUC và độ tin cậy trung bình nếu model hỗ trợ xác suất
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)
        roc = roc_auc_score(y_test, prob[:, 1])
        mean_conf = float(np.mean(np.max(prob, axis=1)))
    else:
        roc = np.nan
        mean_conf = float(acc)  # fallback: dùng Accuracy làm proxy độ tin cậy

    return {
        "name": name,
        "accuracy": float(acc),
        "recall_poisonous": float(recall_poisonous),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc) if not np.isnan(roc) else np.nan,
        "confidence": float(mean_conf),
        "confusion": confusion_matrix(y_test, y_pred),
        "model": model
    }

def print_table(results):
    print("\n=== BẢNG SO SÁNH GIẢI THUẬT ===")
    print(f"{'Giải thuật':35s} | {'Acc':6s} | {'Recall(độc)':12s} | {'F1':6s} | {'ROC-AUC':8s} | {'MeanConf':8s}")
    print("-" * 95)
    for r in results:
        roc_str = f"{r['roc_auc']:.3f}" if not np.isnan(r['roc_auc']) else "N/A"
        print(f"{r['name']:35s} | {r['accuracy']*100:6.2f} | {r['recall_poisonous']*100:12.2f} | "
              f"{r['f1_macro']*100:6.2f} | {roc_str:8s} | {r['confidence']:8.3f}")
    print("-" * 95)

# =========================
# 5. CHƯƠNG TRÌNH CHÍNH
# =========================
def main():
    X_train, X_test, y_train, y_test, encoders, scaler, target_mapping = load_and_prepare()
    models = build_models()
    results = []

    for name, model in models.items():
        print(f"\n🔹 Huấn luyện giải thuật: {name}")
        model.fit(X_train, y_train)

        report = evaluate_model(name, model, X_test, y_test)
        results.append(report)

        print(f"✔ Accuracy                        : {report['accuracy']*100:.2f}%")
        print(f"✔ Recall (lớp nấm độc)            : {report['recall_poisonous']*100:.2f}%")
        print(f"✔ F1-score (macro)                : {report['f1_macro']*100:.2f}%")
        print(f"✔ ROC-AUC                         : {report['roc_auc'] if not np.isnan(report['roc_auc']) else 0:.3f}")
        print(f"✔ Độ tin cậy TB (Mean confidence) : {report['confidence']:.3f}")
        print("Ma trận nhầm lẫn:\n", report['confusion'])
        print("Báo cáo phân loại:\n",
              classification_report(y_test, model.predict(X_test), target_names=["edible(0)", "poisonous(1)"]))

    # In bảng tổng hợp
    print_table(results)

    # Chọn mô hình tốt nhất – ưu tiên Recall nấm độc, sau đó F1, ROC-AUC, Accuracy
    best = sorted(
        results,
        key=lambda r: (r['recall_poisonous'], r['f1_macro'], (r['roc_auc'] if not np.isnan(r['roc_auc']) else -1), r['accuracy']),
        reverse=True
    )[0]

    print("\n🏆 MÔ HÌNH TỐT NHẤT ĐƯỢC LỰA CHỌN:")
    print(f"👉 {best['name']}")
    print(f"   Recall(độc): {best['recall_poisonous']*100:.2f}% | F1: {best['f1_macro']*100:.2f}% | "
          f"ROC-AUC: {best['roc_auc'] if not np.isnan(best['roc_auc']) else 0:.3f} | Acc: {best['accuracy']*100:.2f}%")

    # Lưu bundle theo format Flask (classifier, scaler, encoders, features, target_mapping)
    model_bundle = {
        "classifier": best['model'],
        "scaler": scaler,
        "encoders": encoders,
        "features": FEATURES,
        "target_mapping": target_mapping
    }
    # joblib.dump(model_bundle, OUTPUT_PKL)
    # print(f"\n💾 Đã lưu mô hình tốt nhất tại: {os.path.join(os.getcwd(), OUTPUT_PKL)}")

if __name__ == "__main__":
    main()
