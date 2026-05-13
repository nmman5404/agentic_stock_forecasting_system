def format_vnd(amount: float) -> str:
    """
    Định dạng số thực thành chuỗi tiền tệ VNĐ.
    Ví dụ: 220000.5 -> '220,000 VNĐ'
    """
    try:
        return f"{int(amount):,} VNĐ"
    except (ValueError, TypeError):
        return str(amount)

def get_current_timestamp() -> str:
    """Trả về thời gian hiện tại chuẩn ISO"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")