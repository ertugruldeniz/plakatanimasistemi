# Plaka Tanıma Sistemi

Bu proje, görüntülerden ve kameradan otomatik plaka tanıma işlemi yapan bir web uygulamasıdır.

## Özellikler

- Kamera veya yüklenen resimlerden plaka tanıma
- Tesseract OCR ile plaka metni okuma
- Gerçek zamanlı video işleme
- Gelişmiş görüntü işleme teknikleri
- Kullanıcı dostu web arayüzü

## Gereksinimler

- Python 3.6+
- OpenCV
- Flask
- NumPy
- Tesseract OCR

## Kurulum

1. Tesseract OCR'yi [buradan](https://github.com/UB-Mannheim/tesseract/wiki) indirin ve kurun.
2. Python bağımlılıklarını yükleyin:
```
pip install -r requirements.txt
```

## Kullanım

Uygulamayı başlatmak için:

```
python basla.py
```

veya doğrudan web uygulamasını çalıştırmak için:

```
python web_app.py
```

Tarayıcınızda `http://localhost:5000` adresine giderek uygulamayı kullanabilirsiniz.

## Dosya Yapısı

- `web_app.py`: Ana web uygulaması ve plaka tanıma işlemleri
- `web_fonksiyonlar.py`: Görüntü işleme yardımcı fonksiyonları
- `basla.py`: Uygulama başlatıcı script
- `templates/`: HTML şablonları
- `Resim/`: Test görselleri ve yüklenen resimler
- `requirements.txt`: Python bağımlılıkları
- `NASIL_CALISTIRILIR.md`: Kurulum ve çalıştırma yönergeleri

## Son Güncelleme

Proje temizlendi ve optimize edildi:
- Gereksiz "plate_photo" klasörü kaldırıldı
- Boş "dataset" klasörü kaldırıldı 
- Gereksiz "plakatanimasistemi-master" klasörü kaldırıldı
- Python önbellek dosyaları (__pycache__) temizlendi
- Boş "train_license_plate_model.py" dosyası kaldırıldı

## Lisans

Bu proje açık kaynak olarak lisanslanmıştır.

## 📝 Proje Tanıtımı

Bu proje, kamera veya yüklenen görüntülerden araç plakalarını gerçek zamanlı olarak tespit edip okuyan bir sistemdir. Gelişmiş görüntü işleme teknikleri ve OCR (Optik Karakter Tanıma) kullanarak, plaka metinlerini yüksek doğrulukla tanımlar ve sadece alfanümerik karakterleri korur.

Sistem, canlı kamera görüntüsünü işleyebilir veya yüklenen plaka görsellerini analiz edebilir. Plakaları başarılı bir şekilde tanıyarak, güvenlik, otopark yönetimi veya trafik denetimi gibi alanlarda kullanılabilir.

**Geliştirici:** Ertuğrul Deniz

## 🔧 Kullanılan Teknolojiler

- **Python 3.x**: Ana programlama dili
- **OpenCV**: Görüntü işleme ve plaka tespiti
- **Tesseract OCR**: Optik karakter tanıma
- **Flask**: Web arayüzü
- **NumPy**: Matematiksel işlemler için

## 🚀 Kurulum Adımları

### Gereksinimler

- Python 3.6 veya üzeri
- Webcam (canlı tanıma için)
- Windows için DirectShow desteği (kamera erişimi için)

### 1. Projeyi İndirin

```bash
git clone https://github.com/ertugruldeniz/plakatanimasistemi.git
cd plakatanimasistemi
```

### 2. Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### 3. Tesseract OCR Kurulumu

**Windows için:**
- [Tesseract OCR indirme sayfasından](https://github.com/UB-Mannheim/tesseract/wiki) en son sürümü indirin
- Kurulumu tamamlayın (önerilen konum: `C:\Program Files\Tesseract-OCR\`)
- Kurulum yolunu `web_app.py` dosyasında `pytesseract.pytesseract.tesseract_cmd` değişkenine otomatik tespit eder

**Linux için:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

### 4. Uygulamayı Başlatın

```bash
python basla.py
# veya doğrudan
python web_app.py
```

Uygulama http://localhost:5000 adresinde çalışacaktır.

## 🔍 Nasıl Çalışır?

Sistem aşağıdaki adımları izleyerek plakaları tespit eder ve okur:

1. **Görüntü Alımı**: Canlı kameradan veya yüklenen dosyadan görüntü alınır
2. **Plaka Tespiti Yöntemleri**:
   - **Gelişmiş Görüntü İşleme**: 
     - Gri tonlamaya dönüştürme
     - Gelişmiş gürültü azaltma (Bilateral Filter - 11, 90, 90 parametreleri)
     - Histogram eşitleme
     - Morfolojik işlemler (7x7 çekirdek)
     - Canny kenar tespiti (100, 200 parametreleri)
     - Geliştirilmiş kontur bulma ve filtreleme
3. **Plaka Analizi**:
   - 4-6 köşeli kapalı şekiller
   - Belirli bir en-boy oranına sahip (1.5-6.0 arası)
   - Minimum alan 1000 piksel, maksimum 25000 piksel
   - Dikdörtgensellik oranı kontrolü
   - En iyi 15 kontur analizi
4. **OCR İşlemi**:
   - Plaka bölgesinin iki farklı yöntemle kırpılması
   - Boş görüntü kontrolü
   - CLAHE kontrast iyileştirme
   - Gürültü giderme için morfolojik açma
   - 3x ölçeklendirme
   - Tesseract OCR ile metin okuma
5. **Son İşlem**:
   - Özel karakterlerin ve boşlukların temizlenmesi
   - Sadece alfanümerik karakterlerin korunması ve büyük harfe dönüştürülmesi
   - Plaka uzunluğu kontrolü (4-10 karakter)
   - Sonuçların görselleştirilmesi

### Geliştirilmiş Görsel Geri Bildirim

- Tespit edilen plaka daha belirgin hale getirildi
- Plaka bölgesi mavi dikdörtgen ile görselleştirildi
- İşlenmiş plaka görüntüsü alt kısımda mini pencerede gösteriliyor
- Kenar tespiti sonuçları sağ alt köşede mini pencerede gösteriliyor
- Hata durumları için daha açıklayıcı mesajlar eklendi

## 📊 Örnek Kullanım

### Canlı Kamera Modu

1. Ana sayfadan "Kamerayı Başlat" butonuna tıklayın
2. Kamera başlatıldığında plaka tespiti otomatik başlar
3. Tespit edilen plaka metni ekranda görüntülenir
4. Plaka bölgesi ve işlenmiş görüntü ayrıca gösterilir

### Resim Yükleme Modu

1. Ana sayfadan "Dosya Seç" ile plaka içeren bir görüntü yükleyin
2. "Yükle" butonuna tıklayın
3. Sonuçlar sayfasında tüm işlem adımları ve tespit edilen plaka görüntülenir

## 💡 Performans İyileştirmeleri

- Windows için DirectShow API'si (cv2.CAP_DSHOW) kullanılarak daha hızlı kamera erişimi sağlandı
- Buffer boyutu minimize edilerek kamera açılışı hızlandırıldı
- Görüntü boyutu 640x480'e optimize edildi
- Kamera FPS'i 15 olarak ayarlandı
- 3 saniyelik timeout süresi eklenerek kamera açılışı güvenilir hale getirildi
- Kontur tespiti için RETR_EXTERNAL yöntemi kullanıldı
- Gelişmiş görüntü işleme parametreleri ile hassasiyet artırıldı

## 🔄 Katkıda Bulunma

Bu projeye katkıda bulunmak için:

1. Bu depoyu forklayın
2. Yeni bir branch oluşturun (`git checkout -b yenilik-ekle`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik: Açıklama'`)
4. Branch'inizi push edin (`git push origin yenilik-ekle`)
5. Pull Request oluşturun

### Geliştirme Fikirleri

- **Gelişmiş Görüntü İşleme Algoritmaları**:
  - Adaptif eşikleme ile değişken ışık koşullarına daha iyi uyum
  - Derin öğrenme tabanlı gürültü azaltma ile daha net plaka görüntüleri
  - HDR (High Dynamic Range) görüntü işleme teknikleri
  - Plaka bölgesi için otomatik kontrast ayarlama

- **OCR Performans İyileştirmeleri**:
  - EasyOCR entegrasyonu ile alternatif OCR çözümü
  - Çoklu OCR motorlarının sonuçlarını birleştiren hibrit yaklaşım
  - Türkçe plaka formatına özel OCR eğitimi ve optimizasyon
  - OCR sonuçları için güven puanı hesaplama

- **Kullanıcı Arayüzü Geliştirmeleri**:
  - Mobil cihaz uyumlu tasarım
  - Gerçek zamanlı analiz sonuçları için grafiksel dashboard
  - Tanımlanan plakaların geçmiş kaydı ve raporlama özellikleri
  - İstatistiksel analiz ve veri görselleştirme araçları

- **Plaka Tanıma İyileştirmeleri**:
  - Gece görüşü için özel görüntü iyileştirme algoritmaları
  - Bulanık/yağmurlu/karlı hava koşulları için özel filtreler
  - Farklı ülke plaka formatları için özelleştirilebilir ayarlar
  - Motosiklet ve özel araçlar için plaka tanıma optimizasyonu

- **Sistem Entegrasyonları**:
  - Veritabanı entegrasyonu ile plaka kayıt ve arama sistemi
  - REST API geliştirilerek diğer sistemlerle entegrasyon
  - Bulut tabanlı depolama ve işleme özellikleri
  - MQTT protokolü ile IoT cihazlarına bağlantı seçenekleri

- **Güvenlik ve Gizlilik Özellikleri**:
  - Plaka verilerinin anonimleştirilmesi
  - GDPR uyumlu veri saklama politikaları
  - Kullanıcı yetkilendirme ve erişim kontrolü
  - Şifrelenmiş veri depolama ve iletişim

## 📞 İletişim

Ertuğrul Deniz
