# Main.py

import cv2
import numpy as np
import os

import KarakterTespitEt
import PlakalariTespitEt
import PossiblePlate

# module level variables ##########################################################################
siyah = (0.0, 0.0, 0.0)
beyaz = (255.0, 255.0, 255.0)
sari = (0.0, 255.0, 255.0)
yesil = (0.0, 255.0, 0.0)
kirmizi = (0.0, 0.0, 255.0)

adimleri_goster = False

###################################################################################################
def main():

    KNN_Ogrenme_basarisi = KarakterTespitEt.KNN_verisi_yukle_KNN_ogren()         # attempt KNN training

    if KNN_Ogrenme_basarisi == False:                               # if KNN training was not successful
        print("\nhata: KNN başarılı uygulanamadı\n")  # show error message
        return                                                          # and exit program
    # end if

    orjinal_resim  = cv2.imread("arac_listesi/10.png")               # open image

    if orjinal_resim is None:                            # if image was not read successfully
        print("\nhata: dosyadan resim okunamadı \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    ihtimal_plaka_listeleri = PlakalariTespitEt.plaka_tespit_et(orjinal_resim)           # detect plates

    ihtimal_plaka_listeleri = KarakterTespitEt.plakada_karakter_tespit_et(ihtimal_plaka_listeleri)        # detect chars in plates

    cv2.imshow("orjinal_resim", orjinal_resim)            # show scene image

    if len(ihtimal_plaka_listeleri) == 0:                          # if no plates were found
        print("\nPlaka tespit edilemedi\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        ihtimal_plaka_listeleri.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        Plaka = ihtimal_plaka_listeleri[0]

        cv2.imshow("imgPlate", Plaka.imgPlate)           # show crop of plate and threshold of plate
        cv2.imshow("imgThresh", Plaka.imgThresh)

        if len(Plaka.strChars) == 0:                     # if no chars were found in the plate
            print("\nkarakter tespit edilemedi.\n\n")  # show message
            return                                          # and exit program
        # end if

        PlakaCevresineKirmiziDortgenCiz(orjinal_resim, Plaka)             # draw red rectangle around plate

        print("\nresimden okunan  plaka = " + Plaka.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")

        resimePlakalariIsle(orjinal_resim, Plaka)           # write license plate text on the image

        cv2.imshow("orjinal_resim", orjinal_resim)                # re-show scene image

        cv2.imwrite("orjinal_resim.png", orjinal_resim)           # write image out to file

    # end if else

    cv2.waitKey(0)					# hold windows open until user presses a key

    return
# end main

###################################################################################################
def PlakaCevresineKirmiziDortgenCiz(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), kirmizi, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), kirmizi, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), kirmizi, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), kirmizi, 2)
# end function

###################################################################################################
def resimePlakalariIsle(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, sari, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()


















