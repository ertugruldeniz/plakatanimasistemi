def kamera():

    import numpy as np
    import cv2
    cap = cv2.VideoCapture(0)  # harici bir kamerada i=0 yerine i=1,2,3..vs kullanabiliriz
    while (True):

        # Çerçeveler halinde görüntü yakalar
        ret, frame = cap.read()

        # Üzerinde işlem yapacağımız çerçeve buraya gelsin
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ##Gürültü Temizleme islemi
        noise_removal = cv2.bilateralFilter(gray, 9, 75, 75)
        # Daha iyi sonuç elde etmek için histogram eşitleme yapıldı
        equal_histogram = cv2.equalizeHist(noise_removal)

        # Dikdörtgen yapı elemanı ile morfolojik açılım
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=15)

        # Görüntü çıkarma (Morph görüntüsünü histogram eşitlenmiş görüntüsünden çıkarmak)
        sub_morp_image = cv2.subtract(equal_histogram, morph_image)

        # Görüntüyü eşikleme
        ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)

        # Canny Edge algılama uygulanması
        canny_image = cv2.Canny(thresh_image, 250, 255)
        # Display Image
        canny_image = cv2.convertScaleAbs(canny_image)

        # Kenarları güçlendirmek için genleşme
        kernel = np.ones((3, 3), np.uint8)

        # Genişletme için çekirdek oluşturma
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

        # Sonuç Çerçeveyi Görüntüleme:
        cv2.imshow('Son_Hal', dilated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q ile çıkış yapabilirsiniz
            break

   

    cap.release()
    cv2.destroyAllWindows()
