import pandas as pd
from vnstock import *
from utils.logger import get_logger

logger = get_logger("DataIngestion")

def fetch_historical_data(symbols: list, start_date: str, end_date: str, data_type: str = "stock") -> dict:
    """
    Lấy dữ liệu lịch sử cho một danh sách mã chứng khoán/chỉ số.
    
    Args:
        symbols: list các mã (VD: ['VIC', 'VHM'] hoặc ['VN30'])
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        data_type: 'stock', 'index', hoặc 'derivative'
        
    Returns:
        dict: Dictionary chứa Pandas DataFrame của từng mã. VD: {'VIC': df, 'VHM': df}
    """
    data_dict = {}
    
    for sym in symbols:
        try:
            logger.info(
                "Data fetch started | symbol=%s | data_type=%s | start_date=%s | end_date=%s",
                sym,
                data_type,
                start_date,
                end_date,
            )
            
            # Gọi hàm của vnstock
            df = stock_historical_data(
                symbol=sym, 
                start_date=start_date, 
                end_date=end_date, 
                resolution="1D", 
                type=data_type
            )
            
            if df is not None and not df.empty:
                # vnstock trả về cột 'time', ta đổi tên thành 'date' cho chuẩn mực
                if 'time' in df.columns:
                    df.rename(columns={'time': 'date'}, inplace=True)
                
                # Ép kiểu dữ liệu ngày tháng và đặt làm Index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Sắp xếp lại theo thời gian từ cũ đến mới
                df.sort_index(inplace=True)
                
                data_dict[sym] = df
                logger.info("Data fetch completed | symbol=%s | rows=%s", sym, len(df))
            else:
                logger.warning("Data fetch returned no rows | symbol=%s", sym)
                
        except Exception as e:
            logger.error("Data fetch failed | symbol=%s | error=%s", sym, str(e))
            
    return data_dict

def get_vingroup_and_context_data(start_date: str, end_date: str) -> dict:
    """
    Hàm tổng hợp: Kéo dữ liệu cổ phiếu Vingroup + Context (VN30, Phái sinh).
    """
    logger.info("Ingestion phase started | source=vnstock")
    
    # 1. Kéo cổ phiếu họ Vingroup
    vingroup_symbols = ["VIC", "VHM", "VRE", "VPL"]
    stock_data = fetch_historical_data(vingroup_symbols, start_date, end_date, data_type="stock")
    
    # 2. Kéo Context: Chỉ số VN30
    vn30_data = fetch_historical_data(["VN30"], start_date, end_date, data_type="index")
    
    # 3. Kéo Context: Phái sinh VN30F1M
    derivative_data = fetch_historical_data(["VN30F1M"], start_date, end_date, data_type="derivative")
    
    # Gộp tất cả vào 1 dictionary duy nhất
    all_data = {**stock_data, **vn30_data, **derivative_data}

    logger.info("Ingestion phase completed | symbols_loaded=%s", len(all_data))
    return all_data
