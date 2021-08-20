# Procesamiento  de imagenes
Esta carpeta contiene el script necesario para realizar el pre-procesamiento de las imágenes de las fichas y obtener el texto contenido en estas mediante un OCR.

## Dependencias
* tesseract 4.1.1
* pytesseract
* openCV
* numpy

## Usos
```
python3 ocr.py -src IMAGES_PATH/ -img_out PROCESSED_IMAGES_PATH/ -txt_out OUTPUT_TEXTS_PATH -g (OPTIONAL)
```
Donde:
* IMAGES_PATH: Ruta donde se encuentran las imagenes a procesar.
* PROCESSED_IMAGES_PATH: Ruta donde se almacenan las imagenes procesadas.

* OUTPUT_TEXTS_PATH: Ruta donde se almacenan los textos obtenidos mediante el OCR tesseract.

* -g Determina si las imagenes procesadas se alamacenaran como imagenes binarias o en escala de grises. (Si el parámetro es omitido, se guardaran de forma binaria).