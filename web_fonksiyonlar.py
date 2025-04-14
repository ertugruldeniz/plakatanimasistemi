import cv2
# Opencv Kütüphanesini Projeme Dahil ediyorum.
import numpy as np
#Numpy kütühanesi dahil etme işlemi // Maskeleme işlemlerinde kullanılacak
import os

def resimAc(sec):
    """Dosyadan resim okuma fonksiyonu.
    
    Argümanlar:
        sec: Resmin dosya yolu. Tam yol veya göreceli yol kullanılabilir.
    
    Dönüş:
        Okunan resim matrisini döndürür.
    """
    # Eğer dosya yolu bir dosya adı (göreceli yol) ise ve Resim klasörüne bakılması gerekiyorsa
    if not os.path.exists(sec) and not os.path.isabs(sec):
        # Komut dosyasının bulunduğu klasörün yolu
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Önce doğrudan Resim klasörüne bak
        resim_klasoru = os.path.join(script_dir, "Resim")
        dosya_yolu = os.path.join(resim_klasoru, sec)
        
        # Eğer dosya varsa, oku
        if os.path.exists(dosya_yolu):
            return cv2.imread(dosya_yolu)
            
        # Bir üst dizinde Resim klasörüne bak (bazı yüklemeler için)
        bir_ust_dizin = os.path.dirname(script_dir)
        resim_klasoru = os.path.join(bir_ust_dizin, "Resim")
        dosya_yolu = os.path.join(resim_klasoru, sec)
        
        if os.path.exists(dosya_yolu):
            return cv2.imread(dosya_yolu)
    
    # Eğer tam yol verilmişse veya göreceli yol doğru şekilde verilmişse
    if os.path.exists(sec):
        return cv2.imread(sec)
        
    # Hiçbir şekilde dosya bulunamadıysa hata fırlat
    raise FileNotFoundError(f"Resim dosyası bulunamadı: {sec}")

# RGB uzayından Gri seviyeli resme dönüş işlemi
def griyecevir(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

# Gürültü azaltıcı yumuşatma işlemi
def gurultuAzalt(img_gray):
    gurultuazalt = cv2.bilateralFilter(img_gray, 9, 75, 75)
    return gurultuazalt

# Daha iyi sonuç elde etmek için histogram eşitleme işlemi yapıyoruz
def histogramEsitleme(gurultuazalt):
    histogram_e = cv2.equalizeHist(gurultuazalt)
    return histogram_e

# Açma İşlemi(Opening):
# Aşındırma ile küçük parçalar yok edildikten sonra dilation ile görüntü tekrar genişletilerek küçük parçaların kaybolması sağlanır.
# Gürültülerin etkisi azaltılır.
def morfolojikIslem(h_esitleme):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morfolojikresim = cv2.morphologyEx(h_esitleme, cv2.MORPH_OPEN, kernel, iterations=15)
    return morfolojikresim

# Resim üzerinde düzensiz bölümleri dengelemek.
# İki resim arasındaki değişiklikleri saptamak için görüntü çıkarma kullanılır.(Image subtraction).
def goruntuCikarma(h_esitleme, morfolojik_resim):
    # Görüntü çıkarma (Morph görüntüsünü histogram eşitlenmiş görüntüsünden çıkarmak)
    gcikarilmisresim = cv2.subtract(h_esitleme, morfolojik_resim)
    return gcikarilmisresim

# Görüntüdeki her pikseli siyah/beyaz piksel ile değiştirir
def goruntuEsikle(goruntucikarma):
    ret, goruntuesikle = cv2.threshold(goruntucikarma, 0, 255, cv2.THRESH_OTSU)
    return goruntuesikle

# Görüntünün kenarlarını algılamak için canny edge kullanılır
def cannyEdge(goruntuesikleme):
    canny_goruntu = cv2.Canny(goruntuesikleme, 250, 255)
    canny_goruntu = cv2.convertScaleAbs(canny_goruntu)
    return canny_goruntu

# Dilatasyon operatörü, kenarları güçlendirmek için kullanılır
def genisletmeIslemi(cannedge_goruntu):
    # Kenarları güçlendirmek için genleşme
    cekirdek = np.ones((3, 3), np.uint8)
    # Genişletme için çekirdek oluşturma
    gen_goruntu = cv2.dilate(cannedge_goruntu, cekirdek, iterations=1)
    return gen_goruntu

def konturIslemi(img, gen_goruntu):
    # Kenarlara dayanan resimdeki Konturları Bulma
    # OpenCV 4.x için güncellenmiş findContours kullanımı
    contours, hierarchy = cv2.findContours(gen_goruntu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # Rakamları alana göre sıralama, böylece sayı plakası ilk 10 konturda olacak
    screenCnt = None
    # kontur dng işlemi
    for c in contours:
        # yaklaşık çizgi belirliyoruz
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # % 6 hata ile yaklaşıklık
        # Yaklaşık konturuzun dört noktası varsa, o zaman
        # ----Plakamızı yaklaşık olarak bulduğumuzu varsayabiliriz.

        if len(approx) == 4:  # Konturu 4 köşeli olarak seçiyoruz
            screenCnt = approx
            break
    
    return screenCnt

# Belirlenen alan dışında kalan yerleri maskeleme
def maskelemeIslemi(img_gray, image, screenCnt):
    mask = np.zeros(img_gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    return new_image

# Plakayı iyileştir
def plakaIyilestir(new_image, mask=None, gray=None):
    """Plaka görüntüsünü iyileştirir.
    
    Argümanlar:
        new_image: İşlenecek görüntü
        mask: Maske (belirtilmediyse None)
        gray: Gri görüntü (belirtilmediyse None)
    
    Dönüş:
        İyileştirilmiş plaka görüntüsü
    """
    try:
        # Eğer yeni görüntü YCrCb modeline dönüştürülebilirse
        y, cr, cb = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb))
        # Histogram eşitleme uygula
        y = cv2.equalizeHist(y)
        # Kanalları birleştir ve geri dönüştür
        return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    except Exception as e:
        print(f"Plaka iyileştirme hatası: {e}")
        # Eğer dönüşüm başarısız olursa, orijinal görüntüyü döndür
        return new_image 