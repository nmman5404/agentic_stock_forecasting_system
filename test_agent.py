from src.agent.graph import build_agent_graph
from dotenv import load_dotenv
import os

# Tải API Key từ file .env
load_dotenv()

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("LỖI: Chưa có GOOGLE_API_KEY trong file .env")
        exit()

    app = build_agent_graph()
    
    # Tạo một trạng thái đầu vào giả định (Giả lập việc model bị lỗi MAPE 15%)
    # Để xem Agent có tự động chuyển hướng đi tìm tin tức và Retrain không!
    initial_state = {
        "ticker": "VIC",
        "forecast_data": {
            "metrics": {"MAPE": 0.15}, # Cố tình set lỗi > 5%
            "forecasts": [
                # Cố tình set lỗi Quantile Crossing (0.1 > 0.5)
                {"step": 1, "q_0.025": 100, "q_0.1": 150, "q_0.5": 140, "q_0.9": 160, "q_0.975": 170} 
            ]
        },
        "retry_count": 0
    }
    
    print("\n🚀 BẮT ĐẦU CHẠY LANGGRAPH PIPELINE...\n")
    
    # Chạy Graph
    final_state = app.invoke(initial_state)
    
    print("\n🎯 KẾT QUẢ CUỐI CÙNG TỪ AGENT STATE:")
    print(f"- Trạng thái Evaluate: {final_state['evaluation_status']}")
    print(f"- Action Retrain: {final_state.get('action_taken', 'N/A')}")
    print(f"- Khuyến nghị cuối: {final_state['final_recommendation']}")