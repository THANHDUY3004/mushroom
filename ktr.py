# import pandas as pd

# # ======================================
# # 1. ĐỌC DỮ LIỆU BAN ĐẦU
# # ======================================
# df = pd.read_csv("secondary_data.csv", sep=';')

# print("THÔNG TIN TỔNG QUAN TẬP DỮ LIỆU")
# print(f"Số dòng (mẫu): {df.shape[0]}")
# print(f"Số cột (thuộc tính): {df.shape[1]}")

# # ======================================
# # 2. XEM 5 DÒNG DỮ LIỆU ĐẦU TIÊN
# # =================================== ===
# print("\n 5 DÒNG DỮ LIỆU BAN ĐẦU")
# print(df.head())

# # ======================================
# # 3. KIỂU DỮ LIỆU CÁC THUỘC TÍNH
# # ======================================
# print("\n KIỂU DỮ LIỆU CÁC CỘT ")
# print(df.dtypes)

# # ======================================
# # 4. KIỂM TRA GIÁ TRỊ KHUYẾT
# # ======================================
# print("\nSỐ LƯỢNG GIÁ TRỊ THIẾU TRÊN MỖI CỘT ")
# print(df.isnull().sum())
# #  KIỂM TRA GIÁ TRỊ GINI
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # 1. ĐỌC DỮ LIỆU
# df = pd.read_csv("secondary_data.csv", sep=';')
# FEATURES = [
#     'cap-diameter',
#     'cap-shape',
#     'cap-color',
#     'stem-height',
#     'stem-width',
#     'season'
# ]
# TARGET = 'class'
# df = df.dropna(subset=FEATURES + [TARGET])

# X = df[FEATURES].copy()
# y = df[TARGET].astype(str)
# # 2. MÃ HÓA DỮ LIỆU
# encoders = {}
# categorical_cols = ['cap-shape', 'cap-color', 'season']
# numeric_cols = ['cap-diameter', 'stem-height', 'stem-width']
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col].astype(str))
#     encoders[col] = le

# label_y = LabelEncoder()
# y_encoded = label_y.fit_transform(y)  # e=0, p=1
# # 3. CHIA TRAIN / TEST
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )
# # 4. HUẤN LUYỆN CÂY QUYẾT ĐỊNH (GINI)
# model_gini = DecisionTreeClassifier(
#     criterion="gini",
#     max_depth=10,
#     random_state=42
# )
# model_gini.fit(X_train, y_train)
# # 5. ĐÁNH GIÁ MÔ HÌNH
# y_pred = model_gini.predict(X_test)
# y_proba = model_gini.predict_proba(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# mean_confidence = np.mean(np.max(y_proba, axis=1))
# conf_matrix = confusion_matrix(y_test, y_pred)
# # Recall riêng cho lớp nấm độc (label = 1)
# recall_poisonous = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
# # 6. IN KẾT QUẢ
# print("CÂY QUYẾT ĐỊNH – TIÊU CHÍ GINI ")
# print(f"Accuracy (Độ chính xác): {accuracy*100:.2f}%")
# print(f"Recall (Nấm độc): {recall_poisonous*100:.2f}%")
# print(f"Mean Confidence (Độ tin cậy TB): {mean_confidence:.3f}")
# print("\nMa trận nhầm lẫn:")
# print(conf_matrix)
# print("\nBáo cáo phân loại:")
# print(classification_report(y_test, y_pred, target_names=["Ăn được", "Có độc"]))
# # #  KIỂM TRA GIÁ TRỊ  Entroppy
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # 1. ĐỌC DỮ LIỆU
# df = pd.read_csv("secondary_data.csv", sep=';')
# FEATURES = [
#     'cap-diameter',
#     'cap-shape',
#     'cap-color',
#     'stem-height',
#     'stem-width',
#     'season'
# ]
# TARGET = 'class'
# df = df.dropna(subset=FEATURES + [TARGET])
# X = df[FEATURES].copy()
# y = df[TARGET].astype(str)
# # 2. MÃ HÓA DỮ LIỆU
# encoders = {}
# categorical_cols = ['cap-shape', 'cap-color', 'season']
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col].astype(str))
#     encoders[col] = le

# label_y = LabelEncoder()
# y_encoded = label_y.fit_transform(y)  # e = 0, p = 1
# # 3. CHIA TRAIN / TEST
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )
# # 4. HUẤN LUYỆN CÂY QUYẾT ĐỊNH (ENTROPY)
# model_entropy = DecisionTreeClassifier(
#     criterion="entropy",
#     max_depth=10,
#     random_state=42
# )
# model_entropy.fit(X_train, y_train)
# # 5. ĐÁNH GIÁ MÔ HÌNH
# y_pred = model_entropy.predict(X_test)
# y_proba = model_entropy.predict_proba(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# mean_confidence = np.mean(np.max(y_proba, axis=1))
# conf_matrix = confusion_matrix(y_test, y_pred)
# # Recall riêng cho lớp nấm độc (label = 1)
# recall_poisonous = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
# # 6. IN KẾT QUẢ
# print("CÂY QUYẾT ĐỊNH – TIÊU CHÍ ENTROPY")
# print(f"Accuracy (Độ chính xác): {accuracy*100:.2f}%")
# print(f"Recall (Nấm độc): {recall_poisonous*100:.2f}%")
# print(f"Mean Confidence (Độ tin cậy TB): {mean_confidence:.3f}")
# print("\nMa trận nhầm lẫn:")
# print(conf_matrix)
# print("\nBáo cáo phân loại:")
# print(classification_report(y_test, y_pred, target_names=["Ăn được", "Có độc"]))
# # #  KIỂM TRA GIÁ TRỊ RƯNG NGẪU NHIÊN
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # 1. ĐỌC DỮ LIỆU
# df = pd.read_csv("secondary_data.csv", sep=';')
# FEATURES = [
#     'cap-diameter',   # Đường kính mũ
#     'cap-shape',      # Hình dạng mũ
#     'cap-color',      # Màu sắc mũ
#     'stem-height',    # Chiều cao thân
#     'stem-width',     # Độ rộng thân
#     'season'          # Mùa vụ
# ]
# TARGET = 'class'      # e = ăn được, p = độc
# df = df.dropna(subset=FEATURES + [TARGET])

# X = df[FEATURES].copy()
# y = df[TARGET].astype(str)
# # 2. MÃ HÓA DỮ LIỆU
# encoders = {}
# categorical_cols = ['cap-shape', 'cap-color', 'season']
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col].astype(str))
#     encoders[col] = le
# label_y = LabelEncoder()
# y_encoded = label_y.fit_transform(y)   # e = 0, p = 1
# # 3. CHIA TRAIN / TEST
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded
# )
# # 4. HUẤN LUYỆN RANDOM FOREST
# rf_model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1
# )
# rf_model.fit(X_train, y_train)
# # 5. ĐÁNH GIÁ MÔ HÌNH
# y_pred = rf_model.predict(X_test)
# y_proba = rf_model.predict_proba(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# # Độ tin cậy trung bình (Mean Confidence)
# mean_confidence = np.mean(np.max(y_proba, axis=1))
# conf_matrix = confusion_matrix(y_test, y_pred)
# # Recall riêng cho lớp nấm độc (label = 1)
# recall_poisonous = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
# # 6. IN KẾT QUẢ
# print("RỪNG NGẪU NHIÊN (RANDOM FOREST)")
# print(f"Accuracy (Độ chính xác): {accuracy*100:.2f}%")
# print(f"Recall (Nấm độc): {recall_poisonous*100:.2f}%")
# print(f"Mean Confidence (Độ tin cậy TB): {mean_confidence:.3f}")
# print("\nMa trận nhầm lẫn:")
# print(conf_matrix)
# print("\nBáo cáo phân loại:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=["Ăn được", "Có độc"]
# ))
# # # #  KIỂM TRA GIÁ TRỊ  Tăng cường dần
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # 1. ĐỌC DỮ LIỆU
# df = pd.read_csv("secondary_data.csv", sep=';')
# FEATURES = [
#     'cap-diameter',   # Đường kính mũ
#     'cap-shape',      # Hình dạng mũ
#     'cap-color',      # Màu sắc mũ
#     'stem-height',    # Chiều cao thân
#     'stem-width',     # Độ rộng thân
#     'season'          # Mùa vụ
# ]
# TARGET = 'class'      # e = ăn được, p = độc
# df = df.dropna(subset=FEATURES + [TARGET])
# X = df[FEATURES].copy()
# y = df[TARGET].astype(str)
# # 2. MÃ HÓA DỮ LIỆU
# encoders = {}
# categorical_cols = ['cap-shape', 'cap-color', 'season']
# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col].astype(str))
#     encoders[col] = le
# label_y = LabelEncoder()
# y_encoded = label_y.fit_transform(y)   # e = 0, p = 1
# # 3. CHIA TRAIN / TEST
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_encoded
# )
# # 4. HUẤN LUYỆN GRADIENT BOOSTING
# gb_model = GradientBoostingClassifier(
#     n_estimators=150,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )
# gb_model.fit(X_train, y_train)
# # 5. ĐÁNH GIÁ MÔ HÌNH
# y_pred = gb_model.predict(X_test)
# y_proba = gb_model.predict_proba(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# # Độ tin cậy trung bình (Mean Confidence)
# mean_confidence = np.mean(np.max(y_proba, axis=1))
# conf_matrix = confusion_matrix(y_test, y_pred)
# # Recall riêng cho lớp nấm độc (label = 1)
# recall_poisonous = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
# # 6. IN KẾT QUẢ
# print("TĂNG CƯỜNG DẦN (GRADIENT BOOSTING)")
# print(f"Accuracy (Độ chính xác): {accuracy*100:.2f}%")
# print(f"Recall (Nấm độc): {recall_poisonous*100:.2f}%")
# print(f"Mean Confidence (Độ tin cậy TB): {mean_confidence:.3f}")
# print("\nMa trận nhầm lẫn:")
# print(conf_matrix)
# print("\nBáo cáo phân loại:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=["Ăn được", "Có độc"]
# ))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================
# 1. ĐỌC DỮ LIỆU
# ======================================
df = pd.read_csv("secondary_data.csv", sep=';')

FEATURES = [
    'cap-diameter',   # Đường kính mũ
    'cap-shape',      # Hình dạng mũ
    'cap-color',      # Màu sắc mũ
    'stem-height',    # Chiều cao thân
    'stem-width',     # Độ rộng thân
    'season'          # Mùa vụ
]
TARGET = 'class'      # e = ăn được, p = độc

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES].copy()
y = df[TARGET].astype(str)

# ======================================
# 2. MÃ HÓA DỮ LIỆU
# ======================================
encoders = {}
categorical_cols = ['cap-shape', 'cap-color', 'season']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

label_y = LabelEncoder()
y_encoded = label_y.fit_transform(y)   # e = 0, p = 1

# ======================================
# 3. CHIA TRAIN / TEST
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ======================================
# 4. HUẤN LUYỆN NAIVE BAYES
# ======================================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# ======================================
# 5. ĐÁNH GIÁ MÔ HÌNH
# ======================================
y_pred = nb_model.predict(X_test)
y_proba = nb_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Độ tin cậy trung bình (Mean Confidence)
mean_confidence = np.mean(np.max(y_proba, axis=1))

conf_matrix = confusion_matrix(y_test, y_pred)

# Recall riêng cho lớp nấm độc (label = 1)
recall_poisonous = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])

# ======================================
# 6. IN KẾT QUẢ
# ======================================
print("XÁC SUẤT BAYES (NAIVE BAYES)")
print(f"Accuracy (Độ chính xác): {accuracy*100:.2f}%")
print(f"Recall (Nấm độc): {recall_poisonous*100:.2f}%")
print(f"Mean Confidence (Độ tin cậy TB): {mean_confidence:.3f}")

print("\nMa trận nhầm lẫn:")
print(conf_matrix)

print("\nBáo cáo phân loại:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ăn được", "Có độc"]
))
