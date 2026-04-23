"""
scripts/auto_ingest_reports.py
Cào bảng số liệu HTML trên CafeF -> Chuyển thành Markdown -> Nạp (Ingest) thẳng vào Vector DB cho AI.
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.rag_tools import ingest_text 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
TEMP_DIR = os.path.join(BASE_DIR, "temp_reports")
os.makedirs(TEMP_DIR, exist_ok=True)

VN_TICKERS = [
    "VNM", "VIC", "VHM", "VRE", "HPG", "TCB", "VPB", "MBB", "BID", "CTG",
    "VCB", "FPT", "MSN", "MWG", "GVR", "SAB", "PLX", "GAS", "POW", "PVD",
    "SSI", "VND", "HDB", "TPB", "ACB", "STB", "SHB", "EIB", "KDC", "DGC",
    "DBC", "REE", "PNJ", "KBC", "NVL", "PDR", "DXG", "KDH", "VPI", "AGR",
]

def crawl_bctc_to_markdown(ticker: str):
    print(f"\n--- 🔎 Đang cào dữ liệu Bảng KQKD của mã: {ticker} ---")
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    url = f"https://s.cafef.vn/bao-cao-tai-chinh/{ticker}/incsta/2023/0/0/0/ket-qua-hoat-dong-kinh-doanh.chn"
    
    try:
        driver.get(url)
        time.sleep(3) 
        
        try:
            rows = driver.find_elements(By.XPATH, '//*[@id="tableContent"]/tbody/tr')
        except:
            print(f"  ⏭️ Không tìm thấy cấu trúc bảng dữ liệu cho {ticker}.")
            return
            
        markdown_content = f"### Báo cáo Kết quả Kinh doanh - Mã {ticker.upper()}\n"
        markdown_content += "*(Lưu ý: Dữ liệu được liệt kê theo các năm gần nhất)*\n\n"
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            row_data = [col.text.strip() for col in cols if col.text.strip()]
            
            if len(row_data) >= 2: 
                chi_tieu = row_data[0]
                so_lieu = " | ".join(row_data[1:])
                markdown_content += f"- **{chi_tieu}**: {so_lieu}\n"
                
        file_path = os.path.join(TEMP_DIR, f"{ticker}_BCTC_Table.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        print(f"  ✅ Đã lưu Text: {file_path}")
        
        print(f"  🧠 Đang nạp dữ liệu vào Vector DB (Chroma)...")
        chunks = ingest_text(
            text=markdown_content, 
            ticker=ticker, 
            source=f"CafeF_KQKD_{ticker}.txt", 
            report_type="financial"
        )
        if chunks > 0:
            print(f"  ✅ Đã nạp thành công {chunks} đoạn văn bản cho {ticker}!")
        else:
            print(f"  ❌ Thất bại: Không nạp được dữ liệu cho {ticker} vào Vector DB.")
        
    except Exception as e:
        print(f"  ⚠️ Lỗi khi cào dữ liệu: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    print(f"Bắt đầu quy trình cào và nạp dữ liệu cho {len(VN_TICKERS)} mã cổ phiếu...")
    for ticker in VN_TICKERS:
        crawl_bctc_to_markdown(ticker)