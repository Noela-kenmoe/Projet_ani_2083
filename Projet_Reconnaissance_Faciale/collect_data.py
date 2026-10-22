import os
import sys
import cv2
import tkinter as tk
from tkinter import messagebox
import subprocess

def obtenir_nom_etudiant():
    """Crée une fenêtre graphique moderne et personnalisée pour saisir le nom."""
    nom_saisi = None

    root = tk.Tk()
    root.title("Enregistrement Étudiant - IA")
    root.geometry("950x760")
    root.configure(bg="#1e1e1e")  # Thème sombre moderne
    root.resizable(False, False)

    # Centrer la fenêtre sur l'écran
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Titre
    title_label = tk.Label(root, text="Nouveau Visage à Enregistrer", font=("Segoe UI", 14, "bold"), fg="#ffffff", bg="#1e1e1e")
    title_label.pack(pady=(20, 10))

    # Instruction
    inst_label = tk.Label(root, text="Entrez le nom complet de l'étudiant :", font=("Segoe UI", 11), fg="#b0b0b0", bg="#1e1e1e")
    inst_label.pack(pady=5)

    # Champ de saisie (Entry) bien grand
    entry = tk.Entry(root, font=("Segoe UI", 14), width=28, bg="#2d2d2d", fg="#ffffff", insertbackground="white", relief="flat")
    entry.pack(pady=10, ipady=5)
    entry.focus()  # Met le curseur directement dedans

    def valider(event=None):
        nonlocal nom_saisi
        val = entry.get().strip()
        if val:
            nom_saisi = val
            root.destroy()
        else:
            messagebox.showerror("Erreur", "Le nom ne peut pas être vide !")

    # Permettre de valider en appuyant sur la touche 'Entrée' du clavier
    entry.bind("<Return>", valider)

    # Bouton de validation stylisé
    btn = tk.Button(root, text="Commencer la capture", font=("Segoe UI", 11, "bold"), bg="#cc00c2", fg="#ffffff", activebackground="#005999", activeforeground="#ffffff", relief="flat", padx=15, pady=8, command=valider)
    btn.pack(pady=15)
    btn = tk.Button(root, text="Commencer la reconnaissance", font=("Segoe UI", 11, "bold"), bg="#007acc", fg="#ffffff", activebackground="#005999", activeforeground="#ffffff", relief="flat", padx=15, pady=8, command=lancer_reconnaisance_faciale)
    btn.pack(pady=15)

    root.mainloop()
    return nom_saisi
def lancer_reconnaisance_faciale():
    try:
        subprocess.Popen([sys.executable, "main.py"])
    except Exception as e :
        print (f"erreur lors du lancement de la reconnaisance faicale : {e}")
    

def collect_student_data(output_base_dir="Projet_Reconnaissance_Faciale/dataset", target_count=30):
    print("=== COLLECTE DE DONNÉES ÉTUDIANT ===")
    
    # Appel de notre belle fenêtre personnalisée
    student_name = obtenir_nom_etudiant()
    
    if not student_name:
        print("[INFO] Opération annulée par l'utilisateur.")
        return

    # Normalisation du nom pour le dossier (remplace les espaces par des underscores)
    folder_name = student_name.replace(" ", "_")
    student_dir = output_base_dir + "/" + folder_name

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

        # 5. Enregistrer les photos du visage
        if key == ord('c'):
            if len(faces) == 0:
                print("[AVERTISSEMENT] Aucun visage détecté ! Rapprochez-vous ou ajustez la lumière.")
            else:
                img_path = os.path.join(student_dir, f"img_{count + 1:02d}.jpg")
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