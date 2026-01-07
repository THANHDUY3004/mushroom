import pandas as pd
import joblib
import os
import platform

def clear_screen():
    """Xóa màn hình console tùy theo hệ điều hành (Windows/Linux/Mac)"""
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def main():
    # Khai báo đường dẫn file dữ liệu và file mô hình
    csv_path = "secondary_data.csv"
    pkl_path = "mushroom_random_forest_model.pkl"

    # --- BƯỚC 1: KIỂM TRA TRẠNG THÁI MÔ HÌNH ---
    clear_screen()
    print("=== HỆ THỐNG KIỂM TRA DỮ LIỆU NHẬP LIỆU ===")
    
    if os.path.exists(pkl_path):
        try:
            # Tải file pkl để kiểm tra thông tin mô hình đã lưu
            bundle = joblib.load(pkl_path)
            model_type = type(bundle['classifier']).__name__
            print(f"✅ Mô hình hiện tại: {model_type}")
            print(f"✅ Các đặc trưng AI yêu cầu: {bundle['features']}")
        except:
            print("⚠️ Cảnh báo: File mô hình bị lỗi hoặc không đúng định dạng.")
    else:
        print("⚠️ Cảnh báo: Chưa tìm thấy file .pkl (Cần chạy train.py trước).")

    # --- BƯỚC 2: ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
    if not os.path.exists(csv_path):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu {csv_path}")
        return

    # Đọc file CSV với dấu phân cách chấm phẩy (;)
    df = pd.read_csv(csv_path, sep=';')

    # Danh sách các cột quan trọng nhất để test trên giao diện Web
    selected_cols = [
        'class', 'cap-diameter', 'cap-shape', 'cap-color', 
        'stem-height', 'stem-width', 'season'
    ]

    while True:
        print("\n" + "—"*70)
        print(" CHẾ ĐỘ HIỂN THỊ DỮ LIỆU TEST (5 ĐỘC & 5 ĂN ĐƯỢC)")
        print("—"*70)
        
        input("Nhấn Enter để lấy mẫu dữ liệu ngẫu nhiên mới (hoặc Ctrl+C để thoát)...")
        clear_screen()

        # --- BƯỚC 3: TRÍCH XUẤT MẪU NGẪU NHIÊN ---
        # Lấy 5 mẫu nấm độc (p - poisonous)
        toxic_samples = df[df['class'] == 'p'][selected_cols].sample(5)
        
        # Lấy 5 mẫu nấm ăn được (e - edible)
        edible_samples = df[df['class'] == 'e'][selected_cols].sample(5)

        # --- BƯỚC 4: HIỂN THỊ LÊN MÀN HÌNH ---
        print("\n💀 DANH SÁCH 5 MẪU NẤM ĐỘC (p) - TEST ĐỘ NHẠY AI:")
        print("-" * 70)
        # to_string(index=False) để ẩn đi số thứ tự dòng trong CSV cho gọn
        print(toxic_samples.to_string(index=False))

        print("\n\n🍴 DANH SÁCH 5 MẪU NẤM ĂN ĐƯỢC (e) - TEST ĐỘ AN TOÀN AI:")
        print("-" * 70)
        print(edible_samples.to_string(index=False))

        print("\n" + "="*70)
        print("HƯỚNG DẪN TEST:")
        print("1. Chọn một dòng bất kỳ ở trên.")
        print("2. Nhập các thông số tương ứng vào Form trên trình duyệt.")
        print("3. Kiểm tra xem AI có dự đoán đúng ký tự trong cột 'class' không.")
        print("="*70)

        # Hỏi người dùng có muốn tiếp tục không
        cont = input("\nBạn có muốn lấy mẫu khác không? (y/n): ").lower()
        if cont != 'y':
            print("👋 Kết thúc chương trình kiểm tra.")
            break
        clear_screen()

if __name__ == "__main__":
    main()