import os
import cv2

def collect_student_data(output_base_dir="Projet_Reconnaissance_Faciale/dataset", target_count=30):
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
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 3. Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    count = 0
    print("\n[INSTRUCTIONS]")
    print(f"- L'application va capturer {target_count} photos de votre visage.")
    print("- Appuyez sur la touche 'c' pour enregistrer une photo lorsqu'un visage est détecté.")
    print("- Bougez légèrement la tête (angles, expressions, éclairages) entre chaque capture.")
    print("- Appuyez sur 'q' pour quitter à tout moment.\n")

    while count < target_count:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la lecture du flux vidéo.")
            break

        # Conversion en niveaux de gris pour la détection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 4. Détecter le visage
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )

        display_frame = frame.copy()

        # Afficher un rectangle autour du visage détecté
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Affichage des informations à l'écran
        info_text = f"Etudiant: {student_name} | Photos: {count}/{target_count}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Appuyez sur 'c' pour capturer | 'q' pour quitter", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Collecte de Donnees - ANI-2083", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 5. Enregistrer environ 30 photos du visage
        if key == ord('c'):
            if len(faces) == 0:
                print("[AVERTISSEMENT] Aucun visage détecté ! Rapprochez-vous ou ajustez la lumière.")
            else:
                img_path = os.path.join(student_dir, f"img_{count + 1:02d}.jpg")
                # Sauvegarde de l'image originale (frame)
                cv2.imwrite(img_path, frame)
                count += 1
                print(f"[{count}/{target_count}] Image enregistrée : {img_path}")

        elif key == ord('q'):
            print("[INFO] Capture interrompue par l'utilisateur.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == target_count:
        print(f"\n[SUCCÈS] Capture terminée avec succès ! {count} images enregistrées dans '{student_dir}'.")

if __name__ == "__main__":
    collect_student_data()