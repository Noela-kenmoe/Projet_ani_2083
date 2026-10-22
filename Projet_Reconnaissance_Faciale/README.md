# Projet de reconnaissance faciale

## 1. Présentation

Ce projet permet de reconnaître des étudiants à partir d'images enregistrées dans une base de données. La webcam capture les visages en direct, puis le programme compare chaque visage détecté aux encodages des visages connus.

Le projet utilise :
- **Python** pour la logique du programme ;
- **OpenCV** pour la webcam et l'affichage ;
- **face_recognition** pour encoder et comparer les visages ;
- **MediaPipe Face Detector** pour détecter les visages ;
- **MediaPipe Face Landmarker** pour afficher les points et les contours du visage.

## 2. Structure du projet

```text
Projet_ani_2083/
├── venv/
├── train.py
├── test.py
├── visages_connus.pkl
└── Projet_Reconnaissance_Faciale/
    ├── dataset/
    │   ├── nom_etudiant_1/
    │   ├── nom_etudiant_2/
    │   └── ...
    └── modeles/
        ├── visages_connus.pkl
```

### Rôle des fichiers et dossiers

- `dataset/` : contient un sous-dossier par étudiant, avec ses photos.
- `train.py` : parcourt les photos et prépare les encodages des visages connus.
- `visages_connus.pkl` : contient les encodages et les noms produits par l'entraînement.
- `test.py` : lance la reconnaissance faciale avec la webcam.
- `venv/` : environnement virtuel Python du projet.

## 3. Prérequis

- Python installé.
- Une webcam.
- Les photos des étudiants dans `dataset/`.
- Un fichier `visages_connus.pkl` créé par le programme d'entraînement.
- Les deux modèles MediaPipe placés dans le dossier `modeles/`.

## 4. Installation

Active l'environnement virtuel depuis le dossier du projet. Sous PowerShell, si l'environnement s'appelle `venv` :

```powershell
.\venv\Scripts\Activate.ps1
```

Installe les bibliothèques nécessaires :

```powershell
python -m pip install opencv-python mediapipe face-recognition numpy
```

Vérifie que les bibliothèques sont disponibles :

```powershell
python -c "import cv2, mediapipe, face_recognition, numpy; print('Bibliothèques disponibles')"
```

L'installation de `face_recognition` peut nécessiter des dépendances supplémentaires, notamment `dlib`, selon l'environnement et la version de Python.

## 5. Préparer la base de données

1. Crée un sous-dossier dans `dataset/` pour chaque étudiant.
2. Place les photos de cet étudiant dans son dossier.
3. Utilise des images nettes, avec le visage visible et suffisamment éclairé.
4. Exécute `train.py` pour recalculer les encodages si tu ajoutes des étudiants ou de nouvelles photos.

Le format précis de `visages_connus.pkl` doit correspondre à celui attendu par `test.py`. Dans la version décrite ici, le programme attend les clés `encodings` et `names`.

## 7. Exécution

Depuis le dossier qui contient `test.py`, avec l'environnement virtuel activé, lance :

```powershell
python test.py
```

La webcam devrait s'ouvrir. Appuie sur **Q** pour quitter.

### Affichage

- Rectangle vert : visage reconnu.
- Rectangle rouge : visage identifié comme `Inconnu`.
- Points et lignes : repères et contours du visage.
- Nom affiché : identité retenue par la comparaison des encodages.
- FPS : estimation du nombre d'images traitées ou affichées par seconde, selon le calcul du programme.

Le score de proximité affiché, s'il est activé dans le code, n'est pas une probabilité réelle que l'identité soit correcte.

## 8. Réglages principaux

Les paramètres suivants peuvent être présents dans `test.py` :

```python
TOLERANCE = 0.40
SCALE = 0.5
PROCESS_EVERY_N_FRAMES = 2
LANDMARK_RADIUS = 1
LANDMARK_THICKNESS = 1
```

- `TOLERANCE` : seuil de distance pour accepter une identité. Une valeur plus faible rend généralement la reconnaissance plus stricte.
- `SCALE` : facteur de réduction de l'image. Une image plus petite peut accélérer les calculs, mais peut diminuer la précision.
- `PROCESS_EVERY_N_FRAMES` : fréquence à laquelle la reconnaissance est calculée. Une valeur de `2` signifie une reconnaissance toutes les deux images.
- `LANDMARK_RADIUS` : rayon des petits points du visage.
- `LANDMARK_THICKNESS` : épaisseur des lignes reliant les points.

Les résultats dépendent de la qualité des photos, de l'éclairage, de l'angle du visage, de la webcam et des performances de l'ordinateur.

## 9. Dépannage

### Le programme ne trouve pas un modèle

Vérifie  que les chemins dans `test.py` correspondent à la structure réelle.

### Le programme ne trouve pas `visages_connus.pkl`

Vérifie que ce fichier se trouve à l'emplacement attendu par `KNOWN_FACES_PATH`. Si nécessaire, exécute `train.py` pour le recréer.

### Un étudiant connu est affiché comme inconnu

- Vérifie que ses photos sont nettes et bien éclairées.
- Vérifie que son dossier contient plusieurs photos utiles.
- Vérifie la valeur de `TOLERANCE`.
- Réentraîne le modèle après avoir ajouté ou modifié des photos.

### Le nom alterne entre un étudiant et « Inconnu »

La reconnaissance peut varier d'une image à l'autre. Une logique de stabilisation sur plusieurs images peut limiter ces changements, mais elle doit être intégrée à la boucle de reconnaissance de manière cohérente.

### La vidéo est lente

- Réduis la taille de l'image traitée avec `SCALE`.
- Augmente `PROCESS_EVERY_N_FRAMES` pour calculer la reconnaissance moins souvent.
- Note que le calcul du maillage MediaPipe peut également prendre du temps.

## 10. Confidentialité

Les photos du dataset et les encodages faciaux sont des données sensibles. Ne partage pas publiquement les photos ou le fichier `visages_connus.pkl` sans autorisation. Informe les personnes concernées de l'utilisation de leurs images et respecte les règles applicables dans ton établissement.

## 11. Résumé

Le projet associe une base de données de visages, un fichier d'entraînement, la détection en temps réel par webcam et l'affichage des repères faciaux. Le fichier `train.py` prépare les encodages, tandis que `test.py` les utilise pour comparer les visages détectés aux identités enregistrées.

## AUTEURS 
- Membres du groupe :
1. KENMOE KAMGANG NOELA HERVANE
2. AKOA NNOMO PIERRE STEVE
3. MOUKEKA SOFO CALEB NATHAN
4. NKADA MATHIEU JUAN
5. TENDA FOPA IGOR LOIC