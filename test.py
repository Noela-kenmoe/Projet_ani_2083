"""import face_recognition
import os

image_path = input("Chemin complet de la photo à tester : ").strip().strip('"')

if not os.path.isfile(image_path):
    print("Erreur : fichier introuvable.")
else:
    image = face_recognition.load_image_file(image_path)
    visages = face_recognition.face_locations(image)

    print("Nombre de visages détectés :", len(visages))

    if len(visages) > 0:
        print("Test réussi : un visage a été détecté.")
    else:
        print("Aucun visage détecté. Essaie une photo plus nette.")"""
import os
import face_recognition
import pickle

# Chemin du dossier contenant les étudiants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(
    BASE_DIR,
    "Projet_Reconnaissance_Faciale",
    "dataset"
)

known_faces = []
known_names = []

print("Dossier du dataset :", KNOWN_FACES_DIR)

# Parcourir chaque dossier d'étudiant
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)

    if not os.path.isdir(person_dir):
        continue

    print("Dossier parcouru :", name)

    # Parcourir les photos de cet étudiant
    for filename in os.listdir(person_dir):
        if filename.startswith('.'):
            continue

        image_path = os.path.join(person_dir, filename)

        try:
            image = face_recognition.load_image_file(image_path)

            encodings = face_recognition.face_encodings(
                image,
                face_recognition.face_locations(
                    image,
                    number_of_times_to_upsample=2,
                    model="hog"
                )
            )

            if len(encodings) > 0:
                known_faces.append(encodings[0])
                known_names.append(name)
            else:
                print("Aucun visage détecté :", filename)

        except Exception as e:
            print(f"Erreur dans {image_path} : {e}")

# Sauvegarder les encodages et les noms
data = {
    "encodings": known_faces,
    "names": known_names
}

output_path = os.path.join(BASE_DIR, "visages_connus.pkl")

with open(output_path, "wb") as f:
    pickle.dump(data, f)

print("\nChargement terminé.")
print("Nombre total de visages :", len(known_faces))
print("Étudiants enregistrés :", sorted(set(known_names)))

for nom in sorted(set(known_names)):
    print(nom, ":", known_names.count(nom), "visage(s)")

print("Modèle sauvegardé dans :", output_path)