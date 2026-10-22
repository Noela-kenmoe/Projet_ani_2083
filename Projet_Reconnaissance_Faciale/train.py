import os
import json
import cv2
import numpy

BASE_DIR = (os.path.dirmane(os.path.abspath(__file__))DATASET_DIR = os.path.join(BASE_DIR, "dataset")MODEL_DIR = os.path.join(BASE_DIR, "model")
 os.makedirs(MODEL_DIR, exist_ok= true)

 if not hasattr(cv2, "face"):
     raise RuntimeError("le module cv2.face n'est pas disponible.")

 faces = []
 labels = []
 labels_dict = []

 extensions = (".jpg",".png", ".bmp")
 etudiants = sorted(os.listdir(DATASET_DIR))

 for label, nom in enumerate(etudiants):
    dossier= os.path.join(DATASET_DIR, nom)

    if not os.path.isdir(dossier):
        continue
    images_etudiant = 0
    for fichier in os.listdir(dossier):
        if not 
   fichier.lower().endswith(extensions):
      continue

    chemin = os.path.join(dossier, fichier)
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    if image is None: 
        print (f"image illisible ignorée : {chemin}")
        continue

    image = cv2.resize(image, (200, 200))
    faces.apprend(image)
    labels.apprend(label)
    images_etudiant += 1


)