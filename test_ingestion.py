from datetime import datetime, timedelta
from src.ingestion.vnstock_api import get_vingroup_and_context_data

if __name__ == "__main__":
    # Lấy dữ liệu từ 2 năm trước cho đến ngày hôm nay
    # Điều này đảm bảo ta có đủ dữ liệu để train LightGBM và tạo Feature (MA, Lag)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")

    print(f"Kéo data từ {start_date} đến {end_date}")
    
    # Chạy hàm kéo data
    data = get_vingroup_and_context_data(start_date, end_date)
    
    # In thử kết quả của một mã để kiểm tra
    for sym, df in data.items():
        print(f"\n--- DỮ LIỆU MẪU CỦA {sym} (5 ngày gần nhất) ---")
        print(df.tail())

    print("\nCác mã đã kéo thành công:", list(data.keys()))