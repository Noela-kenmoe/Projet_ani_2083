
import os
import pickle
import cv2
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR,"model", "visages_connus.pkl")

print("Chargement du modèle de visages...")
TOLERANCE = 0.40
FRAME_THICKNESS = 2
FONT_THICKNESS = 2

# Chargement du fichier pickle contenant les visages entraînés
if os.path.exists(MODEL_FILE):
  with open(MODEL_FILE, "rb") as f:
    data = pickle.load(f)
    known_faces = data["encodings"]
    known_names = data["names"]
  print(f"[SUCCES] {len(known_faces)} visages connus chargés depuis le modèle.")
else:
  print(
      f"[ERREUR] Le fichier {MODEL_FILE} est introuvable. Lancez d'abord"
      " train.py !"
  )  
  exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print("Erreur : Impossible d'accéder à la webcam.")
  exit()

print("\n[INFO] Caméra active. Appuyez sur 'q' pour quitter.\n")

# Variables pour l'optimisation de la vitesse (saut de trame)
process_this_frame = True
face_locations = []
face_names = []
face_scores = []
# variablespour stabiliser la reconnaissance
last_name = "None"
consecutive_matches = 0

while True:
  ret, frame = cap.read()
  if not ret:
    print("Erreur lors de la lecture du flux vidéo.")
    break

  # On alterne une frame sur deux pour fluidifier considérablement l'affichage
  if process_this_frame:
    # Réduction de l'image à 1/4 (0.25) pour un calcul ultra-rapide
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Détection des visages avec HOG
    face_locations = face_recognition.face_locations(
        rgb_small_frame, model="hog"
    )
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations
    )

    face_names = []
    face_scores = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
      name = "Inconnu"
      score = 0.0

      if known_faces:
        # Calcul des distances mathématiques avec les visages connus
        
        distances = face_recognition.face_distance(known_faces, face_encoding)
        best_index = distances.argmin()
        best_distance = float(distances[best_index])

        # Calcul d'un pourcentage de proximité indicatif0
        
        # Si la distance est inférieure à la tolérance, c'estq un visage connu
        
        if best_distance < TOLERANCE:
          name = known_names[best_index]
          score = max(0, min(100.0, (1.0 - best_distance) * 100.0))
        else :
          name = "Inconnu"
          score = 0.0
      """
      if best_distance < TOLERANCE:
        detected_name = known_names[best_index]
        if detected_name == last_name:
          consecutive_matches += 1
        else :
          last_name = detected_name
          consecutive_matches = 1
      # confirmer l'identité après 3 reconnaissance consécutives 
      if consecutive_matches >= 2:
        name = detected_name
      else :
        name = "Inconnu"
        score = max(0, min(100.0, (1.0 - best_distance) * 100.0))
    else :
      name = "Inconnu"
      last_name = "None"
      consecutive_matches = 0
      score = 0.0
    """
      face_names.append(name)
      face_scores.append(score)

  process_this_frame = not process_this_frame

  # Affichage des résultats sur l'image de taille réelle
  if(
    isinstance(face_locations, list)
    and isinstance(face_names, list)
    and isinstance(face_scores, list)):
    for face_loc, name, score in zip(face_locations, face_names, face_scores):
      if len(face_loc)== 4 :
         top, right, bottom, left = face_loc
         top *= 4
         right *= 4
         bottom *= 4
         left *= 4

         if name != "Inconnu":
            color = (0, 255, 0)  # Vert pour un visage connu
            label = f"{name} ({score:.1f}%)"
         else:
            color = (0, 0, 255)  # Rouge pour un inconnu
            label = f"I   nconnu ({score:.1f}%)"
         cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
         cv2.putText(
         frame, label, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, FONT_THICKNESS,) 

  # Afficher le flux vidéo en direct
  cv2.imshow("Reconnaissance Faciale en Direct", frame)

  # Quitter avec la touche 'q'
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()

