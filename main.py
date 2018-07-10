import islem

import kamera

print("""        
##############################################                                            # 
#                                            #
#                                            #
#----------- PLAKA TANIMA SİSTEMİNE----------#
#                                            #
#                                            #
##############################################""")

print("Resimden plaka tanıma işlemi yapmak için 1'e")
print("Kamera'dan plaka tanıma işlemi yapmak için 2'ye")

try:
    secim = int(input("Seçiminiz:"))

    if secim == 1:
        islem.goruntu()
    elif secim == 2:
        kamera.kamera()
    else:
        exit()

except ValueError:
    print("Hatalı seçim Lütfen menüden seçim yapınız.")

