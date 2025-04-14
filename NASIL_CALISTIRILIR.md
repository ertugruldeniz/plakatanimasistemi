# Plaka Tanıma Sistemi - Nasıl Çalıştırılır?

Bu dosya, Plaka Tanıma Sistemi web uygulamasını nasıl çalıştıracağınızı açıklar.

## Gereksinimler

Aşağıdaki yazılımların kurulu olması gerekir:

- Python 3.6 veya üzeri 
- Gerekli Python paketleri:
  - Flask
  - OpenCV (cv2)
  - NumPy
  - Pytesseract (isteğe bağlı, OCR işlevi için)
- Tesseract OCR (isteğe bağlı, plaka metnini okumak için)

## Kurulum

### 1. Python'un kurulu olduğundan emin olun

- Windows: [Python indirme sayfası](https://www.python.org/downloads/windows/)
- Linux: Çoğu dağıtımda Python varsayılan olarak kurulur. Değilse, paket yöneticinizi kullanın:
  ```
  # Ubuntu/Debian
  sudo apt-get install python3 python3-pip
  
  # Fedora
  sudo dnf install python3 python3-pip
  ```

### 2. Gerekli paketleri yükleyin

```bash
pip install flask opencv-python numpy pytesseract
```

### 3. Tesseract OCR'yi yükleyin (isteğe bağlı)

- Windows: [Tesseract-OCR indirme sayfası](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: 
  ```
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  
  # Fedora
  sudo dnf install tesseract
  ```

## Uygulamayı Çalıştırma

### Windows

1. `basla.bat` dosyasına çift tıklayın
   
   VEYA
   
2. Komut istemini açın ve şu komutu çalıştırın:
   ```
   python basla.py
   ```

### Linux

1. `basla.sh` dosyasını çalıştırılabilir yapın ve çalıştırın:
   ```
   chmod +x basla.sh
   ./basla.sh
   ```
   
   VEYA
   
2. Terminali açın ve şu komutu çalıştırın:
   ```
   python3 basla.py
   ```

## Erişim

Uygulama başlatıldıktan sonra web tarayıcınızda şu adresi açın:

```
http://localhost:5000
```

## Sorun Giderme

- **"Port 5000 zaten kullanımda" hatası alıyorsanız:**
  Başka bir uygulama bu portu kullanıyor olabilir. Diğer uygulamaları kapatıp tekrar deneyin.

- **Tesseract hatası alıyorsanız:**
  Tesseract OCR kurulu değil veya düzgün yapılandırılmamış. Uygulama yine de çalışacak, ancak plaka metni okuma işlevi çalışmayacaktır.

- **Modül bulunamadı hatası alıyorsanız:**
  Eksik Python paketlerinin kurulu olduğundan emin olun:
  ```
  pip install flask opencv-python numpy pytesseract
  ```

## Not

Uygulama başlatıldığında, web tarayıcınız otomatik olarak açılacaktır. Açılmazsa, manuel olarak "http://localhost:5000" adresini açın. 