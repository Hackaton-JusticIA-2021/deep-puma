import os
import pytesseract as ocr
import cv2 as cv
import numpy as np
import argparse
from tqdm import tqdm

# Definición de parámetros de ejecución #
ap = argparse.ArgumentParser()
ap.add_argument('-src', '--source', type=str, help='ruta a carpeta con imagenes a procesar')
ap.add_argument('-img_out', '--image_out', type=str, help='ruta donde se almacenan las imagenes procesadas')
ap.add_argument('-txt_out', '--text_out', type=str, help='ruta donde se almacenan los textos obtenidos por el OCR')
ap.add_argument('-g', '--gray', default=True, action='store_false',help='dermina si se usa imagen binaria o en escala de grises')
args = vars(ap.parse_args())

# Lectura de parámetros de ejecución #
images_path = args['source']
processed_path = args['image_out']
texts_path = args['text_out']
bin_option = args['gray']

source = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
print('\n[INFO] se encontraron {} imagenes para procesar.\n'.format(len(source)))

for index in tqdm(range(len(source))):
    # Lectura y escalado de imagenes #
    image_name = source[index].split('.')[0]
    image = cv.imread('{}{}'.format(images_path,source[index]))
    image = cv.resize(image, None, fx=1.3, fy=1.3, interpolation=cv.INTER_CUBIC)
    """
    Preprocesamiento de imagenes
    """
    # Eliminación de ruido causado por sombras en las imagenes #
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dilated_image = cv.dilate(gray_image, np.ones((7,7), np.uint8))
    blur_image = cv.medianBlur(dilated_image,21)
    diff_image = 255 - cv.absdiff(gray_image, blur_image)
    norm_image = diff_image.copy()
    cv.normalize(diff_image, norm_image, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    thr_img = cv.threshold(norm_image, 230, 0, cv.THRESH_TRUNC)[1]
    cv.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    bin_image = cv.threshold(thr_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    
    # detección de texto #
    bin_image_inv = cv.threshold(thr_img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)[1]
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (90, 90))
    dilation = cv.dilate(bin_image_inv, rect_kernel, iterations = 1)
    
    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if bin_option:
        image_copy = bin_image.copy()
    else:
        image_copy = gray_image.copy()
    count = -1
    for cnt in contours:
        count +=1
        x,y,w,h = cv.boundingRect(cnt)
        cropped = image_copy[y:y + h, x:x + w]
        h,w = cropped.shape
        if h>490 and w>490:
            #filename = '{}.png'.format('crop_bord'+str(count))
            #cv.imwrite(filename,cropped)
            #text = ocr.image_to_string(Image.open(filename), lang='spa')
            text = ocr.image_to_string(cropped, lang='spa')
            if len(text) > 0:
                cv.imwrite('{}{}_processed_{}.png'.format(processed_path,image_name,count), cropped)
                textfile = open('{}{}_text.txt'.format(texts_path,image_name),'w')
                textfile.write(text)
                textfile.close()
                
