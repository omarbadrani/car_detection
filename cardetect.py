import cv2
import pytesseract
import numpy as np
from PIL import Image
import imutils
import time
import sqlite3
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue

# Configuration du chemin Tesseract pour Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.init_database()

    def init_database(self):
        """Initialise la base de données SQLite pour stocker les plaques détectées"""
        self.conn = sqlite3.connect('license_plates.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Vérifier si la table existe déjà
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detected_plates'")
        table_exists = self.cursor.fetchone()

        if table_exists:
            # Vérifier si la colonne source_type existe
            self.cursor.execute("PRAGMA table_info(detected_plates)")
            columns = [column[1] for column in self.cursor.fetchall()]

            if 'source_type' not in columns:
                # Ajouter la colonne manquante
                self.cursor.execute("ALTER TABLE detected_plates ADD COLUMN source_type TEXT")
                self.conn.commit()
                print("Colonne source_type ajoutée à la table existante.")
        else:
            # Créer la table si elle n'existe pas
            self.cursor.execute('''
                CREATE TABLE detected_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT,
                    detection_time DATETIME,
                    image_path TEXT,
                    source_type TEXT
                )
            ''')
            self.conn.commit()

    def save_to_database(self, plate_text, image_path, source_type="caméra"):
        """Sauvegarde la plaque détectée dans la base de données"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                "INSERT INTO detected_plates (plate_text, detection_time, image_path, source_type) VALUES (?, ?, ?, ?)",
                (plate_text, current_time, image_path, source_type)
            )
            self.conn.commit()
            print(f"Plaque sauvegardée: {plate_text} à {current_time} (Source: {source_type})")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde en base de données: {e}")
            return False

    def get_all_plates(self):
        """Récupère toutes les plaques de la base de données"""
        try:
            self.cursor.execute("SELECT * FROM detected_plates ORDER BY detection_time DESC")
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
            return []


class LicensePlateDetector:
    def __init__(self, db_manager):
        # Créer le dossier pour sauvegarder les plaques détectées
        if not os.path.exists('detected_plates'):
            os.makedirs('detected_plates')

        self.db_manager = db_manager

    def preprocess_image(self, image):
        """Prétraite l'image pour améliorer la détection de texte"""
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Réduction du bruit
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Detection des bords
        edged = cv2.Canny(gray, 170, 200)

        # Trouver les contours dans l'image des bords, puis ne garder que les plus grands
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        return gray, contours

    def detect_license_plate(self, frame):
        """Détecte et reconnaît les plaques d'immatriculation dans une image"""
        try:
            # Redimensionner l'image pour un traitement plus rapide
            frame = imutils.resize(frame, width=800)
            original = frame.copy()

            # Prétraitement de l'image
            gray, contours = self.preprocess_image(frame)

            plate = None
            for contour in contours:
                # Approximation du contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Si le contour a 4 coins, c'est probablement une plaque
                if len(approx) == 4:
                    plate = approx
                    x, y, w, h = cv2.boundingRect(contour)

                    # Vérifier le ratio de la plaque (les plaques ont généralement un ratio largeur/hauteur spécifique)
                    aspect_ratio = w / h
                    if 2 <= aspect_ratio <= 5:
                        break

            if plate is not None:
                # Extraire la région de la plaque
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [plate], 0, 255, -1)
                new_image = cv2.bitwise_and(frame, frame, mask=mask)

                # Extraire les coordonnées de la plaque
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

                # Appliquer un seuil pour améliorer la reconnaissance OCR
                _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # OCR sur l'image recadrée
                try:
                    text = pytesseract.image_to_string(thresh,
                                                       config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                except Exception as e:
                    print(f"Erreur OCR: {e}")
                    text = ""

                if text.strip():
                    # Nettoyer le texte détecté
                    text = ''.join(e for e in text if e.isalnum()).upper()

                    # Dessiner un rectangle autour de la plaque et afficher le texte
                    cv2.drawContours(frame, [plate], -1, (0, 255, 0), 3)
                    cv2.putText(frame, text, (topy, topx - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Sauvegarder l'image et les données
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"detected_plates/plate_{timestamp}_{text}.jpg"
                    cv2.imwrite(image_path, thresh)

                    return frame, text, image_path

            return frame, None, None
        except Exception as e:
            print(f"Erreur dans detect_license_plate: {e}")
            return frame, None, None

    def process_video(self, video_path, progress_queue):
        """Traite une vidéo pour détecter les plaques d'immatriculation"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
                progress_queue.put(("error", "Impossible d'ouvrir la vidéo"))
                return False

            # Obtenir les propriétés de la vidéo
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            print(f"Traitement de la vidéo: {os.path.basename(video_path)}")
            print(f"Durée: {duration:.2f} secondes, {total_frames} images")

            frame_count = 0
            detection_count = 0
            last_detection_time = 0
            detection_cooldown = 2  # Secondes entre deux détections

            # Calculer le saut d'images pour optimiser le traitement
            frame_skip = max(1, int(fps / 5))  # Traiter 5 images par seconde maximum

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Sauter des images pour optimiser le traitement
                if frame_count % frame_skip != 0:
                    continue

                # Détecter les plaques d'immatriculation
                processed_frame, plate_text, image_path = self.detect_license_plate(frame)

                # Sauvegarder si une plaque est détectée
                current_time = frame_count / fps
                if plate_text and (current_time - last_detection_time) > detection_cooldown:
                    self.db_manager.save_to_database(plate_text, image_path, f"vidéo: {os.path.basename(video_path)}")
                    detection_count += 1
                    last_detection_time = current_time

                # Mettre à jour la progression
                if frame_count % (frame_skip * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    progress_queue.put(("progress", progress, detection_count))

            cap.release()

            progress_queue.put(("complete", detection_count))
            print(f"Traitement terminé. {detection_count} plaques détectées dans la vidéo.")
            return True
        except Exception as e:
            print(f"Erreur dans process_video: {e}")
            progress_queue.put(("error", str(e)))
            return False

    def start_surveillance(self, camera_index=0, progress_queue=None, stop_event=None):
        """Démarre la surveillance avec la caméra"""
        try:
            cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                print("Erreur: Impossible d'accéder à la caméra")
                # Essayer avec d'autres index de caméra
                for i in range(1, 5):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        print(f"Caméra trouvée à l'index {i}")
                        camera_index = i
                        break
                else:
                    print("Aucune caméra trouvée. Vérifiez la connexion.")
                    if progress_queue:
                        progress_queue.put(("error", "Aucune caméra trouvée"))
                    return

            print("Surveillance démarrée. Appuyez sur 'q' pour quitter.")

            last_detection_time = 0
            detection_cooldown = 5  # Secondes entre deux détections
            detection_count = 0

            # Créer une fenêtre pour afficher le flux vidéo
            cv2.namedWindow('Surveillance de Plaques d\'Immatriculation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Surveillance de Plaques d\'Immatriculation', 800, 600)

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Erreur: Impossible de lire le flux vidéo")
                    break

                # Détecter les plaques d'immatriculation
                processed_frame, plate_text, image_path = self.detect_license_plate(frame)

                # Afficher le flux vidéo avec les détections
                cv2.imshow('Surveillance de Plaques d\'Immatriculation', processed_frame)

                # Sauvegarder si une plaque est détectée
                current_time = time.time()
                if plate_text and (current_time - last_detection_time) > detection_cooldown:
                    self.db_manager.save_to_database(plate_text, image_path)
                    last_detection_time = current_time
                    detection_count += 1

                    if progress_queue:
                        progress_queue.put(("detection", detection_count))

                # Vérifier si l'utilisateur veut quitter
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            if progress_queue:
                progress_queue.put(("complete", detection_count))

        except Exception as e:
            print(f"Erreur dans start_surveillance: {e}")
            if progress_queue:
                progress_queue.put(("error", str(e)))
            try:
                cv2.destroyAllWindows()
            except:
                pass


class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Détection de Plaques d'Immatriculation")
        self.root.geometry("500x400")

        # Créer le gestionnaire de base de données
        self.db_manager = DatabaseManager()
        self.detector = LicensePlateDetector(self.db_manager)

        self.processing = False
        self.surveillance_running = False
        self.stop_surveillance_event = threading.Event()

        # File d'attente pour la communication entre les threads
        self.progress_queue = queue.Queue()

        self.create_widgets()

        # Démarrer la vérification périodique de la file d'attente
        self.check_progress_queue()

    def create_widgets(self):
        # Titre
        title_label = tk.Label(self.root, text="Détection de Plaques d'Immatriculation",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Frame pour les boutons principaux
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Bouton pour démarrer la surveillance caméra
        self.camera_btn = tk.Button(button_frame, text="Démarrer Surveillance Caméra",
                                    command=self.toggle_camera, height=2, width=30)
        self.camera_btn.pack(pady=10)

        # Bouton pour importer une vidéo
        video_btn = tk.Button(button_frame, text="Importer et Traiter une Vidéo",
                              command=self.import_video, height=2, width=30)
        video_btn.pack(pady=10)

        # Bouton pour afficher les résultats
        results_btn = tk.Button(button_frame, text="Afficher les Résultats",
                                command=self.show_results, height=2, width=30)
        results_btn.pack(pady=10)

        # Frame pour la progression
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(pady=10, fill=tk.X, padx=20)

        self.progress_label = tk.Label(progress_frame, text="Prêt")
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.detection_label = tk.Label(progress_frame, text="Plaques détectées: 0")
        self.detection_label.pack()

        # Bouton pour quitter
        quit_btn = tk.Button(self.root, text="Quitter",
                             command=self.quit_app, height=2, width=30)
        quit_btn.pack(pady=10)

        # Label de statut
        self.status_label = tk.Label(self.root, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def check_progress_queue(self):
        """Vérifie périodiquement la file d'attente pour les mises à jour de progression"""
        try:
            while True:
                try:
                    message = self.progress_queue.get_nowait()
                    self.handle_progress_message(message)
                except queue.Empty:
                    break
        finally:
            # Planifier la prochaine vérification
            self.root.after(100, self.check_progress_queue)

    def handle_progress_message(self, message):
        """Traite les messages de progression de la file d'attente"""
        msg_type = message[0]

        if msg_type == "progress":
            progress, detection_count = message[1], message[2]
            self.progress_bar['value'] = progress
            self.detection_label.config(text=f"Plaques détectées: {detection_count}")
            self.status_label.config(text=f"Traitement en cours: {progress:.1f}%")
        elif msg_type == "detection":
            detection_count = message[1]
            self.detection_label.config(text=f"Plaques détectées: {detection_count}")
        elif msg_type == "complete":
            detection_count = message[1]
            self.processing = False
            self.status_label.config(text="Traitement terminé")
            self.detection_label.config(text=f"Plaques détectées: {detection_count}")
            messagebox.showinfo("Succès", f"Traitement terminé. {detection_count} plaques détectées.")
        elif msg_type == "error":
            error_msg = message[1]
            self.processing = False
            self.status_label.config(text="Erreur")
            messagebox.showerror("Erreur", error_msg)

    def toggle_camera(self):
        if self.surveillance_running:
            self.stop_surveillance_event.set()
            self.camera_btn.config(text="Démarrer Surveillance Caméra")
            self.status_label.config(text="Surveillance arrêtée")
            self.surveillance_running = False
        else:
            if self.processing:
                messagebox.showwarning("Attention", "Un traitement est déjà en cours.")
                return

            self.processing = True
            self.surveillance_running = True
            self.stop_surveillance_event.clear()
            self.camera_btn.config(text="Arrêter Surveillance Caméra")
            self.status_label.config(text="Surveillance caméra en cours...")
            self.detection_label.config(text="Plaques détectées: 0")
            self.root.update()

            # Démarrer dans un thread séparé
            thread = threading.Thread(target=self.run_camera)
            thread.daemon = True
            thread.start()

    def run_camera(self):
        self.detector.start_surveillance(
            progress_queue=self.progress_queue,
            stop_event=self.stop_surveillance_event
        )
        self.processing = False
        self.surveillance_running = False
        self.status_label.config(text="Surveillance caméra terminée")
        self.camera_btn.config(text="Démarrer Surveillance Caméra")

    def import_video(self):
        if self.processing:
            messagebox.showwarning("Attention", "Un traitement est déjà en cours.")
            return

        file_path = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[("Fichiers vidéo", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("Tous les fichiers", "*.*")]
        )

        if file_path:
            self.processing = True
            self.status_label.config(text=f"Traitement de la vidéo: {os.path.basename(file_path)}")
            self.progress_bar['value'] = 0
            self.detection_label.config(text="Plaques détectées: 0")
            self.root.update()

            # Démarrer le traitement dans un thread séparé
            thread = threading.Thread(target=self.process_video_in_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
        else:
            self.status_label.config(text="Aucune vidéo sélectionnée")

    def process_video_in_thread(self, file_path):
        self.detector.process_video(file_path, self.progress_queue)

    def show_results(self):
        # Créer une nouvelle fenêtre pour afficher les résultats
        results_window = tk.Toplevel(self.root)
        results_window.title("Plaques d'Immatriculation Détectées")
        results_window.geometry("800x500")

        # Récupérer les données de la base de données
        try:
            rows = self.db_manager.get_all_plates()

            # Créer un cadre avec une barre de défilement
            frame = tk.Frame(results_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Utiliser un Text widget au lieu de Listbox pour un meilleur affichage
            text_widget = tk.Text(frame, yscrollcommand=scrollbar.set, wrap=tk.WORD, width=90, height=20)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar.config(command=text_widget.yview)

            # Ajouter les données au texte
            if rows:
                text_widget.insert(tk.END, "Plaques d'Immatriculation Détectées\n")
                text_widget.insert(tk.END, "=" * 60 + "\n\n")

                for row in rows:
                    text_widget.insert(tk.END, f"Date/Heure: {row[2]}\n")
                    text_widget.insert(tk.END, f"Plaque: {row[1]}\n")
                    text_widget.insert(tk.END, f"Source: {row[4] if len(row) > 4 else 'N/A'}\n")
                    text_widget.insert(tk.END, f"Image: {row[3]}\n")
                    text_widget.insert(tk.END, "-" * 40 + "\n\n")
            else:
                text_widget.insert(tk.END, "Aucune plaque détectée pour le moment")

            # Empêcher l'édition du texte
            text_widget.config(state=tk.DISABLED)

            # Bouton pour exporter les résultats
            export_btn = tk.Button(results_window, text="Exporter les Résultats",
                                   command=self.export_results)
            export_btn.pack(pady=10)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les résultats: {e}")

    def export_results(self):
        # Exporter les résultats en fichier texte
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )

        if file_path:
            try:
                rows = self.db_manager.get_all_plates()

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Plaques d'Immatriculation Détectées\n")
                    f.write("=" * 50 + "\n\n")

                    for row in rows:
                        f.write(f"Date/Heure: {row[2]}\n")
                        f.write(f"Plaque: {row[1]}\n")
                        f.write(f"Source: {row[4] if len(row) > 4 else 'N/A'}\n")
                        f.write(f"Image: {row[3]}\n")
                        f.write("-" * 30 + "\n")

                messagebox.showinfo("Succès", f"Résultats exportés vers: {file_path}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'exporter les résultats: {e}")

    def quit_app(self):
        self.stop_surveillance_event.set()
        self.root.quit()


# Lancer l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()