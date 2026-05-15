import pandas as pd
from pathlib import Path
from vnstock import Quote
from utils.logger import get_logger

logger = get_logger("DataIngestion")
RAW_CSV_DIR = Path("data/raw/csv")

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
            df = Quote(source="VCI", symbol=sym, show_log=False).history(
                start=start_date,
                end=end_date,
                interval="1D",
            )
            
            if df is not None and not df.empty:
                # vnstock trả về cột 'time', ta đổi tên thành 'date' 
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

def get_market_data(target_ticker: str, start_date: str, end_date: str) -> dict:
    """
    Kéo dữ liệu cổ phiếu mục tiêu + Context (VN30, Phái sinh).
    """
    logger.info("Ingestion phase started | source=vnstock | target_ticker=%s", target_ticker)
    
    # 1. Kéo đúng cổ phiếu mục tiêu
    stock_data = fetch_historical_data([target_ticker], start_date, end_date, data_type="stock")
    
    # 2. Kéo Context: Chỉ số VN30
    vn30_data = fetch_historical_data(["VN30"], start_date, end_date, data_type="index")
    
    # 3. Kéo Context: Phái sinh VN30F1M
    derivative_data = fetch_historical_data(["VN30F1M"], start_date, end_date, data_type="derivative")
    
    # Gộp tất cả vào 1 dictionary duy nhất
    all_data = {**stock_data, **vn30_data, **derivative_data}
    save_raw_csv_snapshots(all_data, RAW_CSV_DIR)

    logger.info("Ingestion phase completed | symbols_loaded=%s", len(all_data))
    return all_data

def save_raw_csv_snapshots(all_data: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for symbol, df in all_data.items():
        if df is None or df.empty:
            logger.warning("Raw CSV snapshot skipped | symbol=%s | reason=empty_dataframe", symbol)
            continue

        csv_df = df.copy()
        if csv_df.index.name == "date" or "date" not in csv_df.columns:
            csv_df = csv_df.reset_index()
        if "date" in csv_df.columns:
            csv_df["date"] = pd.to_datetime(csv_df["date"]).dt.strftime("%Y-%m-%d")
        csv_df["ticker"] = symbol

        preferred_columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
        ordered_columns = [col for col in preferred_columns if col in csv_df.columns]
        ordered_columns.extend(col for col in csv_df.columns if col not in ordered_columns)
        csv_df = csv_df[ordered_columns]

        path = output_dir / f"{symbol}_raw.csv"
        csv_df.to_csv(path, index=False)
        saved_paths.append(path)

    if saved_paths:
        logger.info(
            "Raw CSV snapshots saved:\n%s", saved_paths
        )
    else:
        logger.warning("Raw CSV snapshots skipped | reason=no_dataframes_saved")

    return saved_paths
