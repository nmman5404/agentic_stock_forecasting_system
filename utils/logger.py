import logging
import os
from datetime import datetime

def get_logger(name="AgenticStockSystem"):
    """
    Khởi tạo và cấu hình Logger.
    Log sẽ được in ra màn hình (Console) và lưu vào file trong thư mục logs/
    """
    logger = logging.getLogger(name)
    
    # Tránh việc add handler nhiều lần nếu gọi lại hàm
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format của log: [Thời gian] - [Tên module] - [Mức độ] - [Nội dung]
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 1. Bắn log ra Console (Terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. Lưu log vào file theo ngày
        # Đảm bảo thư mục logs/ tồn tại
        os.makedirs("logs", exist_ok=True)
        log_filename = datetime.now().strftime("%Y-%m-%d") + "_system.log"
        file_handler = logging.FileHandler(os.path.join("logs", log_filename), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
