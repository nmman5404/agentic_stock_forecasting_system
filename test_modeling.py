from src.processing.db_manager import load_from_sqlite
from src.modeling.predictor import generate_7_day_forecast
import json

if __name__ == "__main__":
    # Đọc dữ liệu đã xử lý của VIC từ DB
    df_vic = load_from_sqlite("processed_VIC")
    
    if df_vic.empty:
        print("Lỗi: Không tìm thấy dữ liệu processed_VIC trong Database.")
    else:
        # Chạy dự báo
        result = generate_7_day_forecast(df_vic)
        
        # In kết quả ra xem
        print("\n--- KẾT QUẢ HOLDOUT METRICS ---")
        print(json.dumps(result["metrics"], indent=4))
        
        print("\n--- KẾT QUẢ DỰ BÁO 7 NGÀY (QUANTILES) ---")
        for f in result["forecasts"]:
            print(f"Step {f['step']}:")
            print(f"  Lower 95% (q_0.025): {f['q_0.025']:,.0f}")
            print(f"  Lower 80% (q_0.1)  : {f['q_0.1']:,.0f}")
            print(f"  Point Forecast (q_0.5) : {f['q_0.5']:,.0f} <-- GIÁ DỰ KIẾN")
            print(f"  Upper 80% (q_0.9)  : {f['q_0.9']:,.0f}")
            print(f"  Upper 95% (q_0.975): {f['q_0.975']:,.0f}")
            print("-" * 30)