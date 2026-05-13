from datetime import datetime, timedelta
from src.ingestion.vnstock_api import get_vingroup_and_context_data
from src.processing.cleaner import process_and_save_data
from src.processing.db_manager import load_from_sqlite

if __name__ == "__main__":
    # 1. Lấy dữ liệu
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
    raw_data = get_vingroup_and_context_data(start_date, end_date)
    
    # 2. Xử lý và Lưu DB
    processed_data = process_and_save_data(raw_data)
    
    # 3. Test đọc lại từ SQLite
    print("\n--- TEST ĐỌC TỪ SQLITE ---")
    df_vic_db = load_from_sqlite("processed_VIC")
    print(df_vic_db.tail())
    
    # 4. In ra danh sách cột xem Feature Engineering có chuẩn không
    print("\nCác cột tính năng (Features) đã tạo cho VIC:")
    print(df_vic_db.columns.tolist())