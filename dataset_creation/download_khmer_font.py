"""
Download Noto Sans Khmer font for proper text rendering
"""
import urllib.request
import os
from pathlib import Path

def download_noto_khmer():
    """Download Noto Sans Khmer Regular font"""
    
    fonts_dir = Path("fonts")
    fonts_dir.mkdir(exist_ok=True)
    
    font_file = fonts_dir / "NotoSansKhmer-Regular.ttf"
    
    if font_file.exists():
        print(f"✓ Font already exists: {font_file}")
        return str(font_file)
    
    print("Downloading Noto Sans Khmer font...")
    
    # Direct download URL from Google Fonts GitHub
    url = "https://github.com/notofonts/khmer/raw/main/fonts/NotoSansKhmer/googlefonts/ttf/NotoSansKhmer-Regular.ttf"
    
    try:
        urllib.request.urlretrieve(url, font_file)
        print(f"✓ Downloaded font to: {font_file}")
        return str(font_file)
    except Exception as e:
        print(f"✗ Error downloading font: {e}")
        print("Please manually download from: https://fonts.google.com/noto/specimen/Noto+Sans+Khmer")
        return None

if __name__ == '__main__':
    download_noto_khmer()
