# Extraction du Total des Reçus par OCR

Ce dépôt contient un script Python pour extraire le montant total des reçus en utilisant la reconnaissance optique de caractères (OCR) avec Tesseract. Le script traite plusieurs images, effectue l'OCR et extrait les montants totaux trouvés sur les reçus.

## Fonctionnalités

- Chargement des images et correction de l'orientation.
- Prétraitement des images (conversion en niveaux de gris, détection des bords, etc.).
- Binarisation et détection des contours.
- Correction de l'inclinaison pour les reçus inclinés.
- Rognage et amélioration du texte.
- OCR utilisant Tesseract.
- Extraction des montants totaux à partir des résultats de l'OCR.
- Traitement de plusieurs images de reçus en mode batch.
- Visualisation des étapes de traitement à l'aide de Matplotlib.

## Prérequis

- Python 3.7 ou supérieur
- OpenCV
- Numpy
- Pytesseract
- Imutils
- Matplotlib
- Pillow (PIL)

## Installation

Clonez le dépôt :

```bash
git clone https://github.com/votre-utilisateur/receipt-ocr-extraction.git
```

Installez les dépendances requises :
```bash
pip install opencv-python numpy pytesseract imutils matplotlib pillow
```

Assurez-vous que Tesseract est installé et accessible. Vous pouvez le télécharger depuis le site officiel. Spécifiez ensuite le chemin vers l'exécutable Tesseract dans votre script :
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
## Utilisation

Placez vos images de reçus dans un répertoire, par exemple ./data/.

Modifiez la liste image_paths dans le script principal pour inclure les chemins vers vos images de reçus.

Exécutez le script :

```bash
python main.py
```
Les montants totaux extraits des reçus seront affichés et sauvegardés dans un dictionnaire.

## Exemple de Code
Voici un extrait des étapes principales du script :

```python
# Chargement et orientation de l'image
image = cv2.imread(image_path)
rotated = orient_vertical(image)

# Prétraitement de l'image
edged = sharpen_edge(rotated)

# Binarisation et détection des contours
binary = binarize(edged, 100)
boxed, largest_cnt = find_receipt_bounding_box(binary, rotated)

# Correction de l'inclinaison
rect, angle = find_tilt_angle(largest_cnt)
tilted, delta = adjust_tilt(boxed, angle)

# Rognage et amélioration du texte
cropped = crop(tilted, largest_cnt)
enhanced = enhance_txt(cropped)

# Reconnaissance optique de caractères (OCR)
ocr_text = perform_ocr(enhanced)

# Extraction du total
total = extract_total(ocr_text)

print(f"Montant total extrait : {total}")
```
