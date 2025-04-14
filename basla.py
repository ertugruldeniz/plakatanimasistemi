#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plaka Tanıma Sistemi Başlatma Dosyası
Her iki işletim sisteminde (Windows ve Linux) çalışacak şekilde tasarlanmıştır.
Web uygulamasını 5000 portunda başlatır.
"""

import os
import sys
import platform
import subprocess
import time
import webbrowser
import socket

def is_port_in_use(port):
    """Belirtilen portun kullanımda olup olmadığını kontrol eder"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result == 0
    except:
        return False

def check_requirements():
    """Gerekli paketlerin yüklü olup olmadığını kontrol eder"""
    missing_packages = []
    try:
        import flask
        print("✅ Flask kurulu.")
    except ImportError:
        missing_packages.append("flask")
        
    try:
        import cv2
        print("✅ OpenCV kurulu.")
    except ImportError:
        missing_packages.append("opencv-python")
        
    try:
        import numpy
        print("✅ NumPy kurulu.")
    except ImportError:
        missing_packages.append("numpy")
    
    # Eksik paket kontrolü
    if missing_packages:
        missing_str = ", ".join(missing_packages)
        print(f"❌ Eksik paketler: {missing_str}")
        print("\nGerekli paketleri yüklemek için şu komutu çalıştırın:")
        print(f"pip install {' '.join(missing_packages)}")
        
        if platform.system().lower() == "windows":
            input("\nDevam etmek için ENTER tuşuna basın...")
        return False
    
    print("✅ Temel gereksinimler karşılanıyor.")
    return True

def start_browser(port=5000, delay=2):
    """Web tarayıcıyı belirli bir gecikme ile başlatır"""
    def _open_browser():
        time.sleep(delay)  # Web sunucusu başlamadan önce bekle
        url = f"http://localhost:{port}"
        print(f"\n🔍 Web tarayıcı açılıyor: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️ Tarayıcı otomatik açılamadı: {e}")
            print(f"⚠️ Lütfen manuel olarak şu adresi açın: {url}")
    
    # Yeni bir thread başlat
    import threading
    browser_thread = threading.Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()

def start_app():
    """Web uygulamasını başlatır"""
    # Mevcut script'in yolunu al
    script_dir = os.path.dirname(os.path.abspath(__file__))
    web_app_path = os.path.join(script_dir, "web_app.py")
    
    # Web app dosyasının varlığını kontrol et
    if not os.path.exists(web_app_path):
        print(f"❌ Hata: {web_app_path} bulunamadı.")
        print("Lütfen uygulamayı doğru klasörden çalıştırdığınızdan emin olun.")
        return False
    
    # Port kontrolü
    port = 5000
    if is_port_in_use(port):
        print(f"⚠️ Port {port} zaten kullanımda!")
        print("Lütfen çalışan diğer uygulamaları kapatın ve tekrar deneyin.")
        print("Örneğin: Flask veya başka bir web sunucusu çalışıyor olabilir.")
        
        if platform.system().lower() == "windows":
            input("Devam etmek için ENTER tuşuna basın...")
        return False
    
    # İşletim sistemini tespit et
    os_name = platform.system().lower()
    print(f"ℹ️ İşletim sistemi: {os_name}")
    print(f"ℹ️ Web uygulaması 5000 portunda çalışacak: http://localhost:{port}")
    
    # Tarayıcıyı başlat
    start_browser(port=port)
    
    # Web uygulamasını başlat
    print("🚀 Plaka Tanıma Web Uygulaması başlatılıyor...")
    print("⚠️ Kapatmak için terminal ekranında CTRL+C tuşlarına basın.")
    
    # Resim klasörünü kontrol et, yoksa oluştur
    resim_klasoru = os.path.join(script_dir, "Resim")
    if not os.path.exists(resim_klasoru):
        os.makedirs(resim_klasoru)
        print(f"✅ Resim klasörü oluşturuldu: {resim_klasoru}")
    
    try:
        # En basit ve en güvenilir yöntem: direkt Python ile çalıştırma
        # Farklı işletim sistemleri için Python komutunu ayarla
        python_cmd = "python" if os_name == "windows" else "python3"
        
        # Çalışma dizinini script dizini olarak ayarla
        os.chdir(script_dir)
        print(f"✅ Çalışma dizini ayarlandı: {script_dir}")
        
        # Web uygulamasını başlat
        os.system(f"{python_cmd} {web_app_path}")
        return True
    except Exception as e:
        print(f"❌ Web uygulaması başlatılırken hata oluştu: {e}")
        print("\n🔍 Sorun giderme:")
        print("1. web_app.py dosyasını kontrol edin.")
        print("2. Flask ve diğer gereksinimlerin kurulu olduğundan emin olun.")
        print("3. Doğrudan terminalde 'python web_app.py' komutunu çalıştırın.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Plaka Tanıma Sistemi - Web Uygulaması Başlatma Aracı")
    print("=" * 60)
    
    # Sistemin uygunluğunu kontrol et
    if check_requirements():
        # Uygulamayı başlat
        start_app()
    else:
        print("\n❌ Gereksinimler karşılanmadığı için uygulama başlatılamadı.") 