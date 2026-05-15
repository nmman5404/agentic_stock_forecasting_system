import pandas as pd
from src.processing.features import generate_technical_features, generate_context_features
from src.processing.db_manager import save_to_sqlite
from utils.logger import get_logger

logger = get_logger("DataProcessing")

def process_and_save_data(raw_data_dict: dict):
    """
    Quy trình làm sạch, thêm tính năng (Feature Engineering) và lưu trữ.
    """
    logger.info("Feature engineering phase started")
    
    # 1. Xử lý Context Data (VN30 và VN30F)
    vn30_features = None
    vn30f_features = None
    
    if "VN30" in raw_data_dict:
        vn30_features = generate_context_features(raw_data_dict["VN30"], prefix="vn30")
        # Lưu raw context vào DB
        save_to_sqlite(raw_data_dict["VN30"], "raw_VN30")
        
    if "VN30F1M" in raw_data_dict:
        vn30f_features = generate_context_features(raw_data_dict["VN30F1M"], prefix="vn30f")
        save_to_sqlite(raw_data_dict["VN30F1M"], "raw_VN30F1M")

    # Merge hai bảng context lại với nhau theo index (date)
    context_combined = pd.DataFrame()
    if vn30_features is not None and vn30f_features is not None:
        context_combined = vn30_features.join(vn30f_features, how='outer')
    
    processed_stocks = {}
    stock_symbols = [sym for sym in raw_data_dict.keys() if sym not in ["VN30", "VN30F1M"]]
    
    # 2. Xử lý từng mã cổ phiếu
    for sym in stock_symbols:
        df = raw_data_dict[sym].copy()
        
        # Lưu raw data vào DB trước khi biến đổi
        save_to_sqlite(df, f"raw_{sym}")
        
        # Tạo technical features
        df_featured = generate_technical_features(df)
        
        # Nhúng Context Features (Thị trường chung) vào cổ phiếu
        if not context_combined.empty:
            df_featured = df_featured.join(context_combined, how='left')
            
        # Xóa các dòng bị NaN do quá trình tính lag/rolling/merge
        initial_len = len(df_featured)
        df_featured.dropna(inplace=True)
        final_len = len(df_featured)
        
        logger.info(
            "Feature engineering completed | symbol=%s | rows_dropped=%s | rows_final=%s",
            sym,
            initial_len - final_len,
            final_len,
        )
        
        # Lưu dữ liệu ĐÃ XỬ LÝ (Processed) vào Database
        save_to_sqlite(df_featured, f"processed_{sym}")
        processed_stocks[sym] = df_featured
        
    logger.info("Feature engineering phase completed | database=data/database.sqlite")
    return processed_stocks
