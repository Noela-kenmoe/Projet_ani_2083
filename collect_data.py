import os
import cv2
import pickle
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "Projet_Reconnaissance_Faciale", "dataset")
MODEL_FILE = os.path.join(BASE_DIR, "visages_connus.pkl")
print("chargementdu medèle de visages...")
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'


if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)
        known_faces = data["encodings"]
        known_names = data["names"]
    print(f"[SUCCES] {len(known_faces)} visages connus chargés depuis le modèle.")
else :
    print(f"[ERREUR] le fichier {MODEL_FILE} est introuvable. Lancez d'abord train.py!")
    exit()
"""def collect_student_data(output_base_dir="Projet_Reconnaissance_Faciale/dataset", target_count=30):
    # 1. Demander le nom de l'étudiant
    print("=== COLLECTE DE DONNÉES ÉTUDIANT ===")
    student_name = input("Entrez le nom complet de l'étudiant : ").strip()
    
    if not student_name:
        print("Erreur : Le nom ne peut pas être vide.")
        return

    # Normalisation du nom pour le dossier (remplace les espaces par des underscores)
    folder_name = student_name.replace(" ", "_")
    student_dir = os.path.join(output_base_dir, folder_name)

    # 2. Créer automatiquement son dossier dans dataset/
    os.makedirs(student_dir, exist_ok=True)
    print(f"[INFO] Dossier créé/vérifié : {student_dir}")

    # Chargeur du détecteur de visage Haar Cascade d'OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)"""

    # 3. Ouvrir la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        exit()
count = 0
print("\n[INSTRUCTIONS]")
#print(f"- L'application va capturer {target_count} photos de votre visage.")
print("- Appuyez sur la touche 'c' pour enregistrer une photo lorsqu'un visage est détecté.")
print("- Bougez légèrement la tête (angles, expressions, éclairages) entre chaque capture.")
print("- Appuyez sur 'q' pour quitter à tout moment.\n")

while True:
    ret, image = cap.read()
    
    # S'assurer que la frame a été lue correctement
    if not ret:
        print("Erreur lors de la lecture du flux vidéo.")
        break

    # Détecter les visages sur l'image de la webcam
    """locations = face_recognition.face_locations(image, model='hog')
    
    # Extraire les encodages des visages trouvés sur le flux en direct
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        # Comparer avec les visages connus chargés depuis le fichier pickle
        match = None

    if known_faces:
        distances = face_recognition.face_distance(
        known_faces, face_encoding
    )

    best_match_index = distances.argmin()

    if distances[best_match_index] < TOLERANCE:
        match = known_names[best_match_index]
        print(f"Trouvé : {match}")

        # Coordonnées du rectangle (top, right, bottom, left)
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

    if match:
            # Si la personne est reconnue
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), FRAME_THICKNESS)
        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
            # Si la personne est inconnue
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), FRAME_THICKNESS)
        cv2.putText(image, "Inconnu", (face_location[3] + 10, face_location[2] + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)"""
    # Réduire l'image pour accélérer la détection
    small_frame = cv2.resize(
        image, (0, 0), fx=0.5, fy=0.5
    )

    rgb_small_frame = cv2.cvtColor(
        small_frame, cv2.COLOR_BGR2RGB
    )

    locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    encodings = face_recognition.face_encodings( rgb_small_frame, locations)

    for face_encoding, face_location in zip(encodings, locations):
        match = "Inconnu"
        score = None

        if known_faces:
            distances = face_recognition.face_distance(known_faces, face_encoding)
            best_index = distances.argmin()
            best_distance = float(distances[best_index])
            # Score indicatif de proximité, pas une probabilité
            score = max(0, min(100, (1 - best_distance) * 100))

            if best_distance < TOLERANCE:
                match = known_names[best_index]

        # Les coordonnées doivent être multipliées par 2
        top, right, bottom, left = [
            coord * 2 for coord in face_location
        ]

        if match != "Inconnu":
            color = (0, 255, 0)
            label = f"{match} - Proximite: {score:.1f}%"
        else:
            color = (0, 0, 255)
            label = "Inconnu"

            if score is not None:
                label += f" (meilleure proximite: {score:.1f}%)"

        cv2.rectangle(
            image, (left, top), (right, bottom),
            color, FRAME_THICKNESS
        )

        cv2.putText(
            image, label, (left, max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, FONT_THICKNESS
        )

    # Afficher l'image dans une fenêtre OpenCV
    cv2.imshow("Reconnaissance Faciale en Direct", image)

    # Sortir de la boucle si on appuie sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Libérer la caméra et fermer les fenêtres
cap.release()
