import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import web_fonksiyonlar as fonk
import base64
from werkzeug.utils import secure_filename
import time
import json
import threading
from queue import Queue

# Tesseract OCR'yi koşullu olarak içe aktarma
try:
    import pytesseract
    
    # Tesseract OCR yolunu kontrol et ve ayarla (Windows'ta gerekli)
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    
    # Dinamik olarak kullanıcı profillerindeki Tesseract kurulumlarını da ara
    if os.name == 'nt':  # Windows işletim sistemi ise
        appdata_paths = [
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Tesseract-OCR', 'tesseract.exe'),
            os.path.join(os.environ.get('APPDATA', ''), 'Tesseract-OCR', 'tesseract.exe')
        ]
        tesseract_paths.extend(appdata_paths)
    else:  # Linux/Mac için
        tesseract_paths.extend([
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract'
        ])
    
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"Tesseract OCR bulundu: {path}")
            break
    
    if not tesseract_found:
        print("Tesseract OCR yolu bulunamadı. Lütfen manuel olarak ayarlayın.")
    
    # Tesseract OCR'nin gerçekten çalışıp çalışmadığını kontrol et
    try:
        # Test et - geçersiz bir görüntüyü işlemeye çalış, hata verirse kurulu değil demektir
        dummy_img = np.zeros((10, 10), dtype=np.uint8)
        pytesseract.image_to_string(dummy_img)
        TESSERACT_AVAILABLE = True
        print("Tesseract OCR başarıyla yüklendi ve çalışıyor.")
    except Exception as e:
        TESSERACT_AVAILABLE = False
        print(f"Tesseract OCR kurulu değil veya çalışmıyor: {e}")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Pytesseract mevcut değil. OCR özellikleri devre dışı.")
except Exception as e:
    TESSERACT_AVAILABLE = False
    print(f"Tesseract OCR yüklenemedi: {e}. OCR özellikleri devre dışı.")

app = Flask(__name__)
app.secret_key = "plakatanimasistemi"
app.config['UPLOAD_FOLDER'] = 'Resim'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Global değişken olarak son tespit edilen plaka
son_tespit_edilen_plaka = None
plaka_durumu = {"plaka_bulundu": False, "plaka_metni": ""}
plaka_durumu_lock = threading.Lock()

# SSE istemcileri için kuyruk
sse_kuyruk = Queue()

# Türkiye plaka formatına uygun düzenleme fonksiyonu
def turkiye_plaka_duzenle(plaka_metni):
    """Okunan plaka metnini düzenler, sadece alfanümerik karakterleri korur"""
    # Boş veya çok kısa metinleri düzeltmeye çalışma
    if not plaka_metni or len(plaka_metni) < 2:
        return plaka_metni
    
    # Sadece alfanümerik karakterleri koru
    plaka_metni = ''.join(c for c in plaka_metni if c.isalnum())
    
    # Tüm karakterleri büyük harfe çevir
    plaka_metni = plaka_metni.upper()
    
    return plaka_metni

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def plaka_tani_resim(resim_adi):
    sonuclar = []  # İşlem adımlarının sonuçlarını saklamak için liste
    plaka_metni = "Plaka okunamadı"  # Varsayılan değer
    
    try:
        img = fonk.resimAc(resim_adi)
        _, img_buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("1-Orjinal Resim", img_base64))
        
        img_gray = fonk.griyecevir(img)
        _, img_buffer = cv2.imencode('.jpg', img_gray)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("2-Griye Donusturme İslemi", img_base64))
        
        gurultuazalt = fonk.gurultuAzalt(img_gray)
        _, img_buffer = cv2.imencode('.jpg', gurultuazalt)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("3-Gürültü Temizleme islemi", img_base64))
        
        h_esitleme = fonk.histogramEsitleme(gurultuazalt)
        _, img_buffer = cv2.imencode('.jpg', h_esitleme)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("4-Histogram esitleme islemi", img_base64))
        
        morfolojik_resim = fonk.morfolojikIslem(h_esitleme)
        _, img_buffer = cv2.imencode('.jpg', morfolojik_resim)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("5-Morfolojik acilim", img_base64))
        
        goruntucikarma = fonk.goruntuCikarma(h_esitleme, morfolojik_resim)
        _, img_buffer = cv2.imencode('.jpg', goruntucikarma)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("6-Goruntu cikarma", img_base64))
        
        goruntuesikleme = fonk.goruntuEsikle(goruntucikarma)
        _, img_buffer = cv2.imencode('.jpg', goruntuesikleme)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("7-Goruntu Esikleme", img_base64))
        
        cannedge_goruntu = fonk.cannyEdge(goruntuesikleme)
        _, img_buffer = cv2.imencode('.jpg', cannedge_goruntu)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("8-Canny Edge", img_base64))
        
        gen_goruntu = fonk.genisletmeIslemi(cannedge_goruntu)
        _, img_buffer = cv2.imencode('.jpg', gen_goruntu)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        sonuclar.append(("9-Genisletme", img_base64))
        
        try:
            screenCnt = fonk.konturIslemi(img.copy(), gen_goruntu)
            
            # konturlu görüntüyü al
            konturlu_goruntu = cv2.drawContours(img.copy(), [screenCnt], -1, (9, 236, 255), 3)
            _, img_buffer = cv2.imencode('.jpg', konturlu_goruntu)
            img_base64 = base64.b64encode(img_buffer).decode('utf-8')
            sonuclar.append(("10-Konturlu Goruntu", img_base64))
            
            yeni_goruntu = fonk.maskelemeIslemi(img_gray, img.copy(), screenCnt)
            _, img_buffer = cv2.imencode('.jpg', yeni_goruntu)
            img_base64 = base64.b64encode(img_buffer).decode('utf-8')
            sonuclar.append(("11-Plaka", img_base64))
            
            # Plakadaki metni tespit etme
            if TESSERACT_AVAILABLE:
                try:
                    # Plaka bölgesini tespit etme
                    mask = np.zeros(img_gray.shape, np.uint8)
                    plaka_mask = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
                    
                    # Plaka bölgesini ayıklama (maske ile orijinal görüntünün gri halini çarp)
                    plaka_goruntu = cv2.bitwise_and(img_gray, img_gray, mask=plaka_mask)
                    
                    # Plaka bölgesinin sınırlarını al
                    (x, y) = np.where(plaka_mask == 255)
                    (top_x, top_y) = (np.min(y), np.min(x))
                    (bottom_x, bottom_y) = (np.max(y), np.max(x))
                    
                    # Plaka bölgesini kırp
                    plaka_kirpilmis = plaka_goruntu[top_y:bottom_y+1, top_x:bottom_x+1]
                    
                    # OCR için görüntü ön işleme
                    if TESSERACT_AVAILABLE and plaka_kirpilmis.size > 0:
                        try:
                            # Görüntü iyileştirme aşamaları - OCR başarısını artırmak için
                            
                            # 1. Gürültü azaltma - Gauss filtresi
                            plaka_blur = cv2.GaussianBlur(plaka_kirpilmis, (3, 3), 0)
                            
                            # 2. Kontrast artırma - CLAHE uygula
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
                            plaka_clahe = clahe.apply(plaka_blur)
                            
                            # 3. Adaptif eşikleme - Otsu ile birlikte
                            _, plaka_threshold = cv2.threshold(plaka_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            
                            # 4. Morfolojik işlemler - gürültü giderme ve karakter netleştirme
                            kernel = np.ones((2, 2), np.uint8)
                            plaka_morph = cv2.morphologyEx(plaka_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
                            kernel = np.ones((3, 3), np.uint8)
                            plaka_morph = cv2.morphologyEx(plaka_morph, cv2.MORPH_CLOSE, kernel, iterations=1)
                            
                            # 5. Keskinleştirme
                            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                            plaka_sharp = cv2.filter2D(plaka_morph, -1, sharpen_kernel)
                            
                            # 6. Daha büyük ölçeklendirme (3x) - OCR için daha fazla bilgi
                            plaka_resized = cv2.resize(plaka_sharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                            
                            # 7. İkinci bir eşikleme - daha net siyah-beyaz sonuç
                            _, plaka_final = cv2.threshold(plaka_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            
                            # Mini görüntüler ekle - işlem adımlarını göstermek için
                            h, w = plaka_kirpilmis.shape[:2]
                            
                            # OCR işlemi için çoklu konfigürasyon dene
                            # Farklı parametrelerle 3 kez dene ve en iyi sonucu al
                            plaka_metinleri = []
                            
                            # Konfigürasyon 1: Standart PSM 7 (tek satır)
                            config1 = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                            text1 = pytesseract.image_to_string(plaka_final, config=config1).strip()
                            if text1: plaka_metinleri.append(text1)
                            
                            # Konfigürasyon 2: PSM 8 (kelime)
                            config2 = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                            text2 = pytesseract.image_to_string(plaka_final, config=config2).strip()
                            if text2: plaka_metinleri.append(text2)
                            
                            # Konfigürasyon 3: PSM 6 (tek blok)
                            config3 = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                            text3 = pytesseract.image_to_string(plaka_final, config=config3).strip()
                            if text3: plaka_metinleri.append(text3)
                            
                            # En uzun metni seç (genellikle en iyi sonuç)
                            plaka_metni = ""
                            if plaka_metinleri:
                                plaka_metni = max(plaka_metinleri, key=len)
                            
                            # Metni temizle ve büyük harfe çevir
                            plaka_metni = ''.join(c for c in plaka_metni if c.isalnum()).upper()
                            
                            # Plaka metni kontrolü
                            if plaka_metni and 4 <= len(plaka_metni) <= 10:
                                # Plaka metnini ekrana belirgin şekilde yaz - daha büyük ve vurgu
                                cv2.putText(konturlu_goruntu, f"PLAKA: {plaka_metni}", (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                                # İkinci kez daha ince çizgiyle çizdirerek vurgu efekti
                                cv2.putText(konturlu_goruntu, f"PLAKA: {plaka_metni}", (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1, cv2.LINE_AA)
                                
                                # Plaka durumunu güncelle
                                with plaka_durumu_lock:
                                    plaka_durumu["plaka_bulundu"] = True
                                    plaka_durumu["plaka_metni"] = plaka_metni
                                    sse_kuyruk.put(json.dumps(plaka_durumu))
                                
                                # İşlenmiş plaka görüntüsünü alt köşeye ekle - daha büyük boyutlu
                                h_mini, w_mini = 80, min(240, plaka_final.shape[1] * 80 // plaka_final.shape[0])
                                plaka_mini = cv2.resize(plaka_final, (w_mini, h_mini))
                                plaka_mini_bgr = cv2.cvtColor(plaka_mini, cv2.COLOR_GRAY2BGR)
                                
                                # Görüntü için siyah arka plan ekle
                                bplan = np.zeros((h_mini + 10, w_mini + 10, 3), dtype=np.uint8)
                                bplan[5:5+h_mini, 5:5+w_mini] = plaka_mini_bgr
                                cv2.putText(bplan, "Plaka", (5, h_mini + 8), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                                
                                # Görüntüyü yerleştir
                                y_offset = konturlu_goruntu.shape[0] - h_mini - 10
                                x_offset = konturlu_goruntu.shape[1] - w_mini - 10
                                konturlu_goruntu[y_offset:y_offset+h_mini+10, x_offset:x_offset+w_mini+10] = bplan
                        except Exception as ocr_error:
                            print(f"OCR hatası: {ocr_error}")
                            plaka_metni = ""
                    else:
                        cv2.putText(konturlu_goruntu, "Plaka algilandi (OCR yok)", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                except Exception as ocr_error:
                    print(f"Plaka işleme hatası: {ocr_error}")
                    plaka_metni = "Plaka okunamadı (OCR hatası)"
            else:
                plaka_metni = "OCR desteklenmiyor"
        except Exception as e:
            print("Hata:", e)
            # Temel hata mesajı oluştur
            hata_bilgisi = str(e)[:100] if str(e) else "Bilinmeyen hata"
            return sonuclar, f"Plaka okunamadı (Hata: {hata_bilgisi})"
        
        return sonuclar, plaka_metni
        
    except Exception as e:
        print("Hata:", e)
        # Temel hata mesajı oluştur
        hata_bilgisi = str(e)[:100] if str(e) else "Bilinmeyen hata"
        return [], f"Plaka okunamadı (Hata: {hata_bilgisi})"

# Video yakalama nesnesi
camera = None

# Kamera videosu için frame oluşturucu
def generate_frames():
    """Kameradan frame yakalayıp plaka tanıma işlemi uygular ve sonuçları stream olarak gönderir"""
    # Kamera bağlantısını başlat
    camera = None
    try:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows için DirectShow kullan
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer boyutunu minimize et
        
        # Kamera kalite ayarları
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Daha yüksek çözünürlük
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Daha yüksek çözünürlük
        camera.set(cv2.CAP_PROP_FPS, 30)  # Daha yüksek FPS
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG codec kullan
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
        camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Parlaklık
        camera.set(cv2.CAP_PROP_CONTRAST, 150)  # Kontrast
        camera.set(cv2.CAP_PROP_SATURATION, 150)  # Doygunluk
        
        # Kamera açılamazsa hata ver
        if not camera.isOpened():
            frame = np.zeros((720, 1280, 3), np.uint8)  # Daha büyük hata ekranı
            cv2.putText(frame, "Kamera acilamadi!", (400, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Daha yüksek JPEG kalitesi
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
            
        # Durum değişkenleri
        last_detection_time = 0
        last_plate_text = ""
        text_display_flag = False
        
        # Kameranın ısınması için birkaç kare bekle
        for _ in range(5):
            camera.read()
            time.sleep(0.1)
        
        while True:
            # Frame yakala
            success, frame = camera.read()
            if not success:
                break
            
            # Görüntü iyileştirme - Isı haritası ve netleştirme uygulanabilir burada
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)  # Yüksek kalite yeniden boyutlandırma
            
            # Görüntü netleştirme
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            frame = cv2.filter2D(frame, -1, sharpen_kernel)
            
            # Her 0.5 saniyede bir plaka tespiti yap (performans için)
            current_time = time.time()
            if current_time - last_detection_time > 0.5:
                last_detection_time = current_time
                
                # Plaka tanıma işlemi
                processed_frame, plate_text = camera_plate_recognition(frame)
                
                # Plaka metni varsa, düzenleme yap
                if plate_text:
                    plate_text = turkiye_plaka_duzenle(plate_text)
                    last_plate_text = plate_text
                    text_display_flag = True
            else:
                # Son tanınan plakayı 3 saniye göster
                processed_frame = frame.copy()
                if text_display_flag and last_plate_text and current_time - last_detection_time < 3:
                    cv2.putText(processed_frame, f"PLAKA: {last_plate_text}", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                else:
                    text_display_flag = False
            
            # Kamera kalitesi bilgisini ekle
            cv2.putText(processed_frame, f"Kamera: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(camera.get(cv2.CAP_PROP_FPS))}FPS", 
                       (10, processed_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Frame'i JPEG formatına dönüştür - yüksek kalite
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_bytes = buffer.tobytes()
            
            # Multipart yanıt oluştur
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
    except Exception as e:
        print(f"Video stream hatası: {str(e)}")
        # Hata mesajı içeren frame oluştur
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, f"Stream hatasi: {str(e)[:30]}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        # Kamera kaynağını serbest bırak
        if camera is not None and camera.isOpened():
            camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Resim yükleme işlemi ve plaka tanıma."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Güvenli dosya adı oluştur
            filename = secure_filename(file.filename)
            
            # Script'in bulunduğu klasörün yolunu al
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Resim klasörü yolu oluştur
            upload_folder = os.path.join(script_dir, app.config['UPLOAD_FOLDER'])
            
            # Klasör yoksa oluştur
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
                
            # Dosya yolunu oluştur
            file_path = os.path.join(upload_folder, filename)
            
            # Dosyayı kaydet
            file.save(file_path)
            
            # Plaka tanıma işlemini başlat
            sonuclar, plaka_metni = plaka_tani_resim(file_path)
            
            if sonuclar:
                return render_template('sonuc.html', sonuclar=sonuclar, plaka_metni=plaka_metni)
            else:
                return render_template('index.html', error="Plaka tanıma işlemi sırasında bir hata oluştu!")
        except Exception as e:
            print(f"Hata: {e}")
            return render_template('index.html', error=f"Dosya işlenirken bir hata oluştu: {str(e)}")
    
    return redirect(request.url)

@app.route('/update_plaka', methods=['POST'])
def update_plaka():
    plaka_metni = request.form.get('plaka_metni', '')
    return redirect(url_for('sonuc_manual', plaka_metni=plaka_metni))

@app.route('/sonuc_manual')
def sonuc_manual():
    plaka_metni = request.args.get('plaka_metni', 'Plaka okunamadı')
    return render_template('sonuc.html', sonuclar=[], plaka_metni=plaka_metni)

# Video akışı rotası
@app.route('/video_feed')
def video_feed():
    """Kamera video akışını sağlayan endpoint"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    try:
        if camera is None:
            # Windows'ta DirectShow API kullan - daha hızlı kamera açılışı
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            # Bağlantı bekleme süresini ayarla (milisaniye)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer boyutunu azalt
            
            # 3 saniye timeout sonrasında kontrole geç
            timeout = time.time() + 3  # 3 saniye timeout
            success = False
            
            while time.time() < timeout:
                success = camera.isOpened()
                if success:
                    break
                time.sleep(0.1)  # Kısa bir bekleme
            
            if not success:
                raise Exception("Kamera açılamadı (zaman aşımı)")
            
            # Kamera optimize ayarları - daha yüksek çözünürlük ve kalite
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Daha yüksek çözünürlük
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Daha yüksek çözünürlük
            camera.set(cv2.CAP_PROP_FPS, 30)  # Daha yüksek FPS
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG codec kullan
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposure
            camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Parlaklık
            camera.set(cv2.CAP_PROP_CONTRAST, 150)  # Kontrast
            camera.set(cv2.CAP_PROP_SATURATION, 150)  # Doygunluk
            
            print("✅ Kamera başarıyla başlatıldı.")
        return redirect(url_for('camera_stream'))
    except Exception as e:
        print(f"❌ Kamera başlatma hatası: {e}")
        return render_template('index.html', error=f"Kamera başlatılamadı: {str(e)}. Lütfen kamera bağlantınızı kontrol edin.")

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return redirect(url_for('index'))

@app.route('/camera_stream')
def camera_stream():
    return render_template('camera.html')

# SSE endpoint - plaka durum bilgisini yayınla
@app.route('/plaka_durumu')
def plaka_durumu_stream():
    def event_stream():
        while True:
            if not sse_kuyruk.empty():
                data = sse_kuyruk.get()
                yield f"data: {data}\n\n"
            time.sleep(0.2)  # CPU kullanımını düşük tutmak için
    
    return Response(event_stream(), mimetype="text/event-stream")

def get_plate_text(plate_img):
    """
    Plaka görüntüsünden Tesseract OCR kullanarak plaka metnini çıkarır.
    
    Args:
        plate_img: İşlenmiş plaka görüntüsü
        
    Returns:
        str: Tanımlanan plaka metni veya tanıma başarısız olduysa boş string
    """
    try:
        if plate_img is None or plate_img.size == 0:
            print("Boş plaka resmi, OCR yapılamıyor.")
            return ""
            
        # Görüntüyü iyileştir
        # Daha büyük boyut (x3) 
        h, w = plate_img.shape[:2]
        plate_img = cv2.resize(plate_img, (w*3, h*3))
        
        # Netleştirme filtreleri uygula
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        plate_img = cv2.filter2D(plate_img, -1, kernel)
        
        # Tesseract OCR yapılandırması
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # OCR uygulaması
        text = pytesseract.image_to_string(plate_img, config=custom_config)
        
        # Temizleme
        text = ''.join(c for c in text if c.isalnum())
        
        # Uygun uzunluk kontrolü (TR plakalar genellikle 5-9 karakter)
        if len(text) < 4 or len(text) > 10:
            return ""
            
        return text
    except Exception as e:
        print(f"OCR hatası: {str(e)}")
        return ""

def camera_plate_recognition(frame):
    """Kamera görüntüsünden plaka tanıma fonksiyonu"""
    try:
        # Görüntü boyutunu işleme göre optimize et
        if frame.shape[0] > 720 or frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        
        # Orijinal frame'i kopyala (işleme sonunda gösterilecek)
        original_frame = frame.copy()
        
        # Resimden Plaka Tanıma algoritmasını uygula
        # 1. Griye dönüştürme - Daha iyi gri tonlama için ağırlıklı dönüşüm
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gürültü azaltma - Geliştirilmiş bilateral filtre
        gurultuazalt = cv2.bilateralFilter(img_gray, 11, 90, 90)
        
        # 3. Histogram eşitleme - CLAHE kullanarak daha iyi kontrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        h_esitleme = clahe.apply(gurultuazalt)
        
        # 4. Morfolojik işlem - Daha büyük çekirdek
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morfolojik_resim = cv2.morphologyEx(h_esitleme, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 5. Görüntü çıkarma
        goruntucikarma = cv2.subtract(h_esitleme, morfolojik_resim)
        
        # 6. Görüntü eşikleme - Adaptif eşikleme
        goruntuesikleme = cv2.adaptiveThreshold(goruntucikarma, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)
        
        # 7. Canny edge - Optimize edilmiş parametreler
        cannedge_goruntu = cv2.Canny(goruntuesikleme, 100, 200)
        
        # 8. Genişletme işlemi - Daha büyük çekirdek, daha fazla iterasyon
        dilation_kernel = np.ones((3, 3), np.uint8)
        gen_goruntu = cv2.dilate(cannedge_goruntu, dilation_kernel, iterations=2)
        
        # 9. Kontur işlemi - resimden plaka tanımaya benzer şekilde
        plaka_bulundu = False
        plaka_metni = ""
        
        try:
            # Kontur tespiti - daha fazla kontur bul
            contours, _ = cv2.findContours(gen_goruntu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Konturları alanlarına göre sırala (büyükten küçüğe)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Debug görüntüsü - kontur sayısını ekle
            debug_info = f"Kontur sayısı: {len(contours)}" 
            cv2.putText(original_frame, debug_info, (10, original_frame.shape[0] - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # En iyi 15 konturu dene
            best_contours = contours[:15] if len(contours) > 15 else contours
            
            for contour in best_contours:
                # Kontur alanı çok küçükse atla
                if cv2.contourArea(contour) < 1000:
                    continue
                    
                # Kontur çevresini hesapla
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Dikdörtgenimsi şekilleri bul (4-6 köşe)
                if 4 <= len(approx) <= 6:
                    # Dikdörtgen kontrolü - min bounding rect
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # En-boy oranı kontrolü (plakalar için 1.5-6.0 arası)
                    aspect_ratio = float(w) / h
                    if 1.5 <= aspect_ratio <= 6.0:
                        # Dikdörtgensellik kontrolü
                        rect_area = w * h
                        contour_area = cv2.contourArea(contour)
                        
                        # Alanlar benzer olmalı
                        rect_similarity = float(contour_area) / rect_area
                        if rect_similarity > 0.7:  # İyi bir dikdörtgen olma olasılığı
                            plaka_bulundu = True
                            screenCnt = approx
                            
                            # Plaka bölgesini mavi dikdörtgen ile göster
                            cv2.drawContours(original_frame, [screenCnt], -1, (255, 0, 0), 3)
                            
                            # Plakadaki metni tespit etme
                            if TESSERACT_AVAILABLE:
                                try:
                                    # Plaka bölgesini tespit etme
                                    mask = np.zeros(img_gray.shape, np.uint8)
                                    plaka_mask = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
                                    
                                    # Plaka bölgesini ayıklama (maske ile orijinal görüntünün gri halini çarp)
                                    plaka_goruntu = cv2.bitwise_and(img_gray, img_gray, mask=plaka_mask)
                                    
                                    # Plaka bölgesinin sınırlarını al
                                    (x, y) = np.where(plaka_mask == 255)
                                    (top_x, top_y) = (np.min(y), np.min(x))
                                    (bottom_x, bottom_y) = (np.max(y), np.max(x))
                                    
                                    # Plaka bölgesini kırp
                                    plaka_kirpilmis = plaka_goruntu[top_y:bottom_y+1, top_x:bottom_x+1]
                                    
                                    # OCR için görüntü ön işleme
                                    if TESSERACT_AVAILABLE and plaka_kirpilmis.size > 0:
                                        try:
                                            # Görüntü iyileştirme aşamaları - OCR başarısını artırmak için
                                            
                                            # 1. Gürültü azaltma - Gauss filtresi
                                            plaka_blur = cv2.GaussianBlur(plaka_kirpilmis, (3, 3), 0)
                                            
                                            # 2. Kontrast artırma - CLAHE uygula
                                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                                            plaka_clahe = clahe.apply(plaka_blur)
                                            
                                            # 3. Adaptif eşikleme - Otsu ile birlikte
                                            _, plaka_threshold = cv2.threshold(plaka_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                            
                                            # 4. Morfolojik işlemler - gürültü giderme ve karakter netleştirme
                                            kernel = np.ones((2, 2), np.uint8)
                                            plaka_morph = cv2.morphologyEx(plaka_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
                                            kernel = np.ones((3, 3), np.uint8)
                                            plaka_morph = cv2.morphologyEx(plaka_morph, cv2.MORPH_CLOSE, kernel, iterations=1)
                                            
                                            # 5. Keskinleştirme
                                            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                                            plaka_sharp = cv2.filter2D(plaka_morph, -1, sharpen_kernel)
                                            
                                            # 6. Daha büyük ölçeklendirme (3x) - OCR için daha fazla bilgi
                                            plaka_resized = cv2.resize(plaka_sharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                                            
                                            # 7. İkinci bir eşikleme - daha net siyah-beyaz sonuç
                                            _, plaka_final = cv2.threshold(plaka_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                            
                                            # OCR işlemi için çoklu konfigürasyon dene
                                            # Farklı parametrelerle 3 kez dene ve en iyi sonucu al
                                            plaka_metinleri = []
                                            
                                            # Konfigürasyon 1: Standart PSM 7 (tek satır)
                                            config1 = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                            text1 = pytesseract.image_to_string(plaka_final, config=config1).strip()
                                            if text1: plaka_metinleri.append(text1)
                                            
                                            # Konfigürasyon 2: PSM 8 (kelime)
                                            config2 = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                            text2 = pytesseract.image_to_string(plaka_final, config=config2).strip()
                                            if text2: plaka_metinleri.append(text2)
                                            
                                            # Konfigürasyon 3: PSM 6 (tek blok)
                                            config3 = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                            text3 = pytesseract.image_to_string(plaka_final, config=config3).strip()
                                            if text3: plaka_metinleri.append(text3)
                                            
                                            # En uzun metni seç (genellikle en iyi sonuç)
                                            plaka_metni = ""
                                            if plaka_metinleri:
                                                plaka_metni = max(plaka_metinleri, key=len)
                                            
                                            # Metni temizle ve büyük harfe çevir
                                            plaka_metni = ''.join(c for c in plaka_metni if c.isalnum()).upper()
                                            
                                            # Plaka metni kontrolü
                                            if plaka_metni and 4 <= len(plaka_metni) <= 10:
                                                # Plaka metnini ekrana belirgin şekilde yaz - daha büyük ve vurgu
                                                cv2.putText(original_frame, f"PLAKA: {plaka_metni}", (10, 50), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                                                # İkinci kez daha ince çizgiyle çizdirerek vurgu efekti
                                                cv2.putText(original_frame, f"PLAKA: {plaka_metni}", (10, 50), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1, cv2.LINE_AA)
                                                
                                                # Plaka durumunu güncelle
                                                with plaka_durumu_lock:
                                                    plaka_durumu["plaka_bulundu"] = True
                                                    plaka_durumu["plaka_metni"] = plaka_metni
                                                    sse_kuyruk.put(json.dumps(plaka_durumu))
                                                
                                                # İşlenmiş plaka görüntüsünü alt köşeye ekle - daha büyük boyutlu
                                                h_mini, w_mini = 80, min(240, plaka_final.shape[1] * 80 // plaka_final.shape[0])
                                                plaka_mini = cv2.resize(plaka_final, (w_mini, h_mini))
                                                plaka_mini_bgr = cv2.cvtColor(plaka_mini, cv2.COLOR_GRAY2BGR)
                                                
                                                # Görüntü için siyah arka plan ekle
                                                bplan = np.zeros((h_mini + 10, w_mini + 10, 3), dtype=np.uint8)
                                                bplan[5:5+h_mini, 5:5+w_mini] = plaka_mini_bgr
                                                cv2.putText(bplan, "Plaka", (5, h_mini + 8), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                                                
                                                # Görüntüyü yerleştir
                                                y_offset = original_frame.shape[0] - h_mini - 10
                                                x_offset = original_frame.shape[1] - w_mini - 10
                                                original_frame[y_offset:y_offset+h_mini+10, x_offset:x_offset+w_mini+10] = bplan
                                                
                                                # Kenar tespiti sonuçlarını sağ alt köşeye ekle
                                                edge_h, edge_w = 80, 120
                                                edge_mini = cv2.resize(cannedge_goruntu, (edge_w, edge_h))
                                                edge_mini_bgr = cv2.cvtColor(edge_mini, cv2.COLOR_GRAY2BGR)
                                                original_frame[original_frame.shape[0]-edge_h:original_frame.shape[0], 
                                                            original_frame.shape[1]-w_mini-edge_w-10:original_frame.shape[1]-w_mini-10] = edge_mini_bgr
                                                
                                                # Plaka bulundu, işlemi sonlandır
                                                break
                                        except Exception as ocr_error:
                                            print(f"OCR hatası: {ocr_error}")
                                            plaka_metni = ""
                                except Exception as ocr_error:
                                    print(f"Plaka işleme hatası: {ocr_error}")
                                    plaka_metni = "Plaka okunamadı (OCR hatası)"
                            else:
                                plaka_metni = "OCR desteklenmiyor"
                                cv2.putText(original_frame, "Plaka algilandi (OCR yok)", (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Kontur işleme hatası: {e}")
            plaka_bulundu = False
        
        # Plaka bulunamadıysa durumu göster
        if not plaka_bulundu:
            cv2.putText(original_frame, "Plaka araniyor...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        # İşleme adımlarını görselleştir - debug için
        # Canny kenar görüntüsünü sağ alt köşede küçük olarak göster
        h_mini, w_mini = 100, 100
        canny_mini = cv2.resize(cannedge_goruntu, (w_mini, h_mini))
        canny_mini_bgr = cv2.cvtColor(canny_mini, cv2.COLOR_GRAY2BGR)
        original_frame[original_frame.shape[0]-h_mini:original_frame.shape[0], 
                       original_frame.shape[1]-w_mini-100:original_frame.shape[1]-100] = canny_mini_bgr
        
        # Genişletilmiş görüntüyü sol alt köşede göster
        gen_mini = cv2.resize(gen_goruntu, (w_mini, h_mini))
        gen_mini_bgr = cv2.cvtColor(gen_mini, cv2.COLOR_GRAY2BGR)
        original_frame[original_frame.shape[0]-h_mini:original_frame.shape[0], 
                       0:w_mini] = gen_mini_bgr
        
        # Gerçek zamanlı durumu göster
        cv2.putText(original_frame, f"Kamera aktif | Resimden Plaka Algılama Modu", 
                   (10, original_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        return original_frame, plaka_metni
        
    except Exception as e:
        print(f"Plaka tanıma hatası: {str(e)}")
        # Hata durumunda orijinal frame'i ve boş plaka metni döndür
        if 'frame' in locals():
            cv2.putText(frame, f"Islem hatasi: {str(e)[:30]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return frame, ""
        else:
            # Son çare: boş bir frame oluştur
            empty_frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(empty_frame, "Kamera hatasi!", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return empty_frame, ""

if __name__ == '__main__':
    # Resim klasörünü kontrol et, yoksa oluştur
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    print(f"OCR durumu: {'Aktif' if TESSERACT_AVAILABLE else 'Devre dışı'}")
    print("Yapay zeka destekli plaka karakter düzeltme aktif.")
    print("Web uygulaması başlatılıyor... http://127.0.0.1:5000 adresinden erişebilirsiniz.")
    
    # Başlangıçta SSE kuyruğuna ilk durumu yolla
    sse_kuyruk.put(json.dumps(plaka_durumu))
    
    app.run(host='0.0.0.0', port=5000, debug=True) 