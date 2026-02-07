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
import re

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

    def preprocess_for_small_images(self, image):
        """Prétraitement spécial pour les petites images"""
        # Agrandir l'image si elle est trop petite
        height, width = image.shape[:2]

        if height < 300 or width < 300:
            # Calculer le facteur d'agrandissement
            scale = max(600 / height, 600 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return image

    def enhance_image_quality(self, image):
        """Améliore la qualité de l'image pour une meilleure détection"""
        # Agrandir si nécessaire
        image = self.preprocess_for_small_images(image)

        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE pour améliorer le contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Réduction du bruit
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.medianBlur(gray, 3)

        # Améliorer la netteté
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)

        return gray

    def find_license_plate_regions(self, image):
        """Trouve les régions potentiellement intéressantes"""
        gray = self.enhance_image_quality(image)

        # Plusieurs méthodes pour trouver les régions de texte

        # Méthode 1: Seuillage adaptatif inversé
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        # Méthode 2: Seuillage Otsu inversé
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Méthode 3: Détection de bords
        edges = cv2.Canny(gray, 50, 150)

        # Combiner les méthodes
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, edges)

        # Opérations morphologiques pour regrouper les régions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.dilate(combined, kernel, iterations=2)
        combined = cv2.erode(combined, kernel, iterations=1)

        # Trouver les contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer et trier les contours
        potential_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Aire minimale
                x, y, w, h = cv2.boundingRect(contour)

                # Ratio largeur/hauteur pour les plaques
                aspect_ratio = w / float(h)
                if 1.2 < aspect_ratio < 6.0:  # Ratio plus large
                    potential_regions.append((contour, x, y, w, h))

        # Trier par aire
        potential_regions.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)

        return gray, potential_regions[:10]  # Prendre les 10 plus grandes régions

    def extract_and_preprocess_plate(self, gray_image, region):
        """Extrait et prétraite une région potentielle de plaque"""
        contour, x, y, w, h = region

        # Ajouter une marge
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(gray_image.shape[1], x + w + margin_x)
        y_end = min(gray_image.shape[0], y + h + margin_y)

        # Extraire la région
        plate_region = gray_image[y_start:y_end, x_start:x_end]

        # Redimensionner pour OCR si nécessaire
        if plate_region.shape[0] < 50:
            scale = 100 / plate_region.shape[0]
            new_width = int(plate_region.shape[1] * scale)
            plate_region = cv2.resize(plate_region, (new_width, 100), interpolation=cv2.INTER_CUBIC)

        # Plusieurs méthodes de seuillage
        # 1. Seuillage adaptatif
        thresh1 = cv2.adaptiveThreshold(plate_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        # 2. Seuillage Otsu
        _, thresh2 = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(plate_region)
        _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Choisir la meilleure par contraste
        images = [thresh1, thresh2, thresh3]
        contrasts = [np.std(img) for img in images]
        best_idx = np.argmax(contrasts)
        best_image = images[best_idx]

        # Nettoyer l'image
        kernel = np.ones((2, 2), np.uint8)
        best_image = cv2.morphologyEx(best_image, cv2.MORPH_CLOSE, kernel)
        best_image = cv2.morphologyEx(best_image, cv2.MORPH_OPEN, kernel)

        # Ajouter une bordure
        best_image = cv2.copyMakeBorder(best_image, 20, 20, 20, 20,
                                        cv2.BORDER_CONSTANT, value=255)

        return best_image, (x_start, y_start, x_end - x_start, y_end - y_start)

    def recognize_text(self, plate_image):
        """Reconnaît le texte sur l'image de plaque"""
        # Configurations OCR pour plaques
        configs = [
            '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]

        best_text = ""
        best_confidence = 0

        for config in configs:
            try:
                data = pytesseract.image_to_data(plate_image, config=config, output_type=pytesseract.Output.DICT)

                # Extraire texte avec bonne confiance
                texts = []
                confidences = []

                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text and int(data['conf'][i]) > 30:  # Seuil bas
                        texts.append(text)
                        confidences.append(int(data['conf'][i]))

                if texts:
                    combined_text = ''.join(texts).replace(' ', '').upper()
                    avg_confidence = sum(confidences) / len(confidences)

                    # Nettoyer
                    cleaned = re.sub(r'[^A-Z0-9]', '', combined_text)

                    # Valider format basique
                    if 4 <= len(cleaned) <= 10 and avg_confidence > best_confidence:
                        best_text = cleaned
                        best_confidence = avg_confidence

            except Exception as e:
                continue

        return best_text if best_text else None

    def detect_license_plate_in_frame(self, frame):
        """Détecte la plaque dans une frame"""
        try:
            # Redimensionner pour traitement
            original = frame.copy()
            frame = imutils.resize(frame, width=1000)  # Agrandir plus

            print(f"Dimensions après redimensionnement: {frame.shape}")

            # Trouver les régions potentielles
            gray, potential_regions = self.find_license_plate_regions(frame)

            best_plate_text = None
            best_bbox = None

            # Essayer chaque région
            for i, region in enumerate(potential_regions):
                print(f"Test région {i + 1}: {region[1:5]}")

                # Extraire et prétraiter
                plate_image, bbox = self.extract_and_preprocess_plate(gray, region)

                # Sauvegarder pour débogage
                debug_path = f"detected_plates/debug_region_{i}.jpg"
                cv2.imwrite(debug_path, plate_image)
                print(f"Région {i + 1} sauvegardée: {debug_path}")

                # Reconnaître texte
                plate_text = self.recognize_text(plate_image)

                if plate_text:
                    print(f"Texte potentiel détecté: {plate_text}")
                    best_plate_text = plate_text
                    best_bbox = bbox
                    break

            # Si détection réussie
            if best_plate_text and best_bbox:
                x, y, w, h = best_bbox

                # Ajuster les coordonnées pour l'image originale
                scale_x = original.shape[1] / frame.shape[1]
                scale_y = original.shape[0] / frame.shape[0]
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)

                # Dessiner sur l'image originale
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(original, best_plate_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Sauvegarder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"detected_plates/plate_{timestamp}_{best_plate_text}.jpg"
                cv2.imwrite(image_path, original[y:y + h, x:x + w])

                return original, best_plate_text, image_path

            print("Aucune plaque détectée après vérification de toutes les régions")
            return frame, None, None

        except Exception as e:
            print(f"Erreur détection: {e}")
            import traceback
            traceback.print_exc()
            return frame, None, None

    def process_image_with_direct_ocr(self, image_path):
        """Approche alternative: OCR direct sur l'image entière"""
        try:
            # Lire image
            image = cv2.imread(image_path)
            if image is None:
                return None, None

            # Agrandir l'image
            height, width = image.shape[:2]
            if height < 400 or width < 400:
                scale = max(800 / height, 800 / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Convertir en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Améliorer le contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Lisser
            gray = cv2.medianBlur(gray, 3)

            # Seuillage
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR sur l'image entière
            config = '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(thresh, config=config)

            # Chercher des motifs de plaque
            patterns = [
                r'\b[A-Z]{2}\d{3}[A-Z]{2}\b',  # Format français: AB123CD
                r'\b\d{4}[A-Z]{2}\d{2}\b',  # Format ancien: 1234AB56
                r'\b\d{3,4}[A-Z]{2,3}\d{0,3}\b',  # Format tunisien approximatif
                r'\b[A-Z]{1,3}\d{3,6}[A-Z]{0,3}\b'  # Format général
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text.upper())
                if matches:
                    # Prendre la meilleure correspondance (la plus longue)
                    best_match = max(matches, key=len)
                    if len(best_match) >= 5:
                        print(f"Plaque trouvée par OCR direct: {best_match}")

                        # Sauvegarder
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f"detected_plates/ocr_direct_{timestamp}_{best_match}.jpg"
                        cv2.imwrite(image_path, image)

                        return best_match, image_path

            return None, None

        except Exception as e:
            print(f"Erreur OCR direct: {e}")
            return None, None

    def process_image_file(self, image_path, progress_queue=None):
        """Traite une image avec plusieurs méthodes"""
        try:
            if not os.path.exists(image_path):
                if progress_queue:
                    progress_queue.put(("error", f"Fichier introuvable: {image_path}"))
                return False

            image = cv2.imread(image_path)
            if image is None:
                if progress_queue:
                    progress_queue.put(("error", f"Impossible de lire l'image: {image_path}"))
                return False

            print(f"\n=== Traitement de l'image: {os.path.basename(image_path)} ===")
            print(f"Dimensions originales: {image.shape}")

            # Méthode 1: Détection par contours
            print("\nMéthode 1: Détection par contours...")
            processed_image, plate_text, saved_path = self.detect_license_plate_in_frame(image)

            if plate_text:
                print(f"SUCCÈS avec méthode 1: {plate_text}")
                self.db_manager.save_to_database(plate_text, saved_path, f"image: {os.path.basename(image_path)}")

                cv2.imshow(f"Détection - {plate_text}", processed_image)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

                if progress_queue:
                    progress_queue.put(("complete", 1, plate_text))
                return True

            # Méthode 2: OCR direct
            print("\nMéthode 2: OCR direct...")
            plate_text, saved_path = self.process_image_with_direct_ocr(image_path)

            if plate_text:
                print(f"SUCCÈS avec méthode 2: {plate_text}")
                self.db_manager.save_to_database(plate_text, saved_path,
                                                 f"image: OCR direct - {os.path.basename(image_path)}")

                # Afficher l'image
                img = cv2.imread(image_path)
                cv2.imshow(f"OCR Direct - {plate_text}", img)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

                if progress_queue:
                    progress_queue.put(("complete", 1, plate_text))
                return True

            # Méthode 3: Découpage manuel
            print("\nMéthode 3: Découpage manuel...")
            result = self.try_manual_cropping(image_path)
            if result:
                plate_text, saved_path = result
                print(f"SUCCÈS avec méthode 3: {plate_text}")
                self.db_manager.save_to_database(plate_text, saved_path,
                                                 f"image: Manuel - {os.path.basename(image_path)}")

                if progress_queue:
                    progress_queue.put(("complete", 1, plate_text))
                return True

            print("\nÉCHEC: Aucune méthode n'a fonctionné")
            message = "Aucune plaque détectée. Essayez avec une image plus grande et plus claire."
            print(message)

            cv2.destroyAllWindows()

            if progress_queue:
                progress_queue.put(("complete", 0, message))
            return False

        except Exception as e:
            error_msg = f"Erreur: {e}"
            print(error_msg)
            if progress_queue:
                progress_queue.put(("error", error_msg))
            cv2.destroyAllWindows()
            return False

    def try_manual_cropping(self, image_path):
        """Permet à l'utilisateur de sélectionner manuellement la plaque"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Agrandir l'image pour sélection
            height, width = image.shape[:2]
            display_image = image.copy()

            if height < 600:
                scale = 600 / height
                new_width = int(width * scale)
                new_height = 600
                display_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Demander à l'utilisateur de sélectionner la plaque
            print("\nSélectionnez la plaque avec la souris (rectangle)")
            print("Cliquez et glissez pour dessiner un rectangle, puis appuyez sur ENTER")

            # Sélection ROI
            roi = cv2.selectROI("Sélectionnez la plaque", display_image, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()

            if roi[2] > 0 and roi[3] > 0:  # Si une sélection valide
                x, y, w, h = roi

                # Ajuster les coordonnées si l'image a été redimensionnée
                if height < 600:
                    scale_factor = height / 600
                    x = int(x * scale_factor)
                    y = int(y * scale_factor)
                    w = int(w * scale_factor)
                    h = int(h * scale_factor)

                # Extraire la région
                plate_region = image[y:y + h, x:x + w]

                if plate_region.size == 0:
                    return None

                # Prétraiter pour OCR
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Agrandir pour OCR
                if thresh.shape[0] < 100:
                    scale = 150 / thresh.shape[0]
                    new_width = int(thresh.shape[1] * scale)
                    thresh = cv2.resize(thresh, (new_width, 150), interpolation=cv2.INTER_CUBIC)

                # OCR
                config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(thresh, config=config)

                # Nettoyer
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

                if len(cleaned) >= 4:
                    # Sauvegarder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_path = f"detected_plates/manual_{timestamp}_{cleaned}.jpg"
                    cv2.imwrite(saved_path, plate_region)

                    return cleaned, saved_path

            return None

        except Exception as e:
            print(f"Erreur découpage manuel: {e}")
            return None

    def process_video(self, video_path, progress_queue):
        """Traite une vidéo"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
                progress_queue.put(("error", "Impossible d'ouvrir la vidéo"))
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Traitement de la vidéo: {os.path.basename(video_path)}")
            print(f"Images totales: {total_frames}")

            frame_count = 0
            detection_count = 0
            frame_skip = max(1, int(fps / 3))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Détecter
                processed_frame, plate_text, image_path = self.detect_license_plate_in_frame(frame)

                if plate_text:
                    self.db_manager.save_to_database(plate_text, image_path, f"vidéo: {os.path.basename(video_path)}")
                    detection_count += 1

                # Mettre à jour progression
                if frame_count % (frame_skip * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    progress_queue.put(("progress", progress, detection_count))

            cap.release()

            progress_queue.put(("complete", detection_count))
            print(f"Traitement terminé. {detection_count} plaques détectées.")
            return True
        except Exception as e:
            print(f"Erreur dans process_video: {e}")
            progress_queue.put(("error", str(e)))
            return False


class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Détection de Plaques - Version Améliorée")
        self.root.geometry("650x600")

        self.db_manager = DatabaseManager()
        self.detector = LicensePlateDetector(self.db_manager)

        self.processing = False
        self.surveillance_running = False
        self.stop_surveillance_event = threading.Event()
        self.progress_queue = queue.Queue()

        self.create_widgets()
        self.check_progress_queue()

    def create_widgets(self):
        # Titre
        title_label = tk.Label(self.root, text="Détection de Plaques d'Immatriculation",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Frame pour les boutons principaux
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Boutons
        self.camera_btn = tk.Button(button_frame, text="Démarrer Surveillance Caméra",
                                    command=self.toggle_camera, height=2, width=35)
        self.camera_btn.pack(pady=5)

        image_btn = tk.Button(button_frame, text="Traiter une Image (Auto)",
                              command=self.import_image, height=2, width=35)
        image_btn.pack(pady=5)

        manual_btn = tk.Button(button_frame, text="Traiter une Image (Manuel)",
                               command=self.import_image_manual, height=2, width=35)
        manual_btn.pack(pady=5)

        video_btn = tk.Button(button_frame, text="Traiter une Vidéo",
                              command=self.import_video, height=2, width=35)
        video_btn.pack(pady=5)

        results_btn = tk.Button(button_frame, text="Afficher les Résultats",
                                command=self.show_results, height=2, width=35)
        results_btn.pack(pady=5)

        # Frame pour les conseils
        tips_frame = tk.LabelFrame(self.root, text="Instructions")
        tips_frame.pack(pady=10, padx=20, fill=tk.X)

        tips_text = """Pour de meilleurs résultats:
1. Utilisez 'Traiter une Image (Auto)' d'abord
2. Si ça échoue, utilisez 'Traiter une Image (Manuel)'
3. Pour mode manuel: tracez un rectangle autour de la plaque
4. Utilisez des images de bonne qualité (> 800x600 pixels)"""

        tips_label = tk.Label(tips_frame, text=tips_text, justify=tk.LEFT)
        tips_label.pack(padx=10, pady=5)

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
                             command=self.quit_app, height=2, width=35)
        quit_btn.pack(pady=10)

        # Label de statut
        self.status_label = tk.Label(self.root, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def check_progress_queue(self):
        try:
            while True:
                try:
                    message = self.progress_queue.get_nowait()
                    self.handle_progress_message(message)
                except queue.Empty:
                    break
        finally:
            self.root.after(100, self.check_progress_queue)

    def handle_progress_message(self, message):
        msg_type = message[0]

        if msg_type == "progress":
            progress, detection_count = message[1], message[2]
            self.progress_bar['value'] = progress
            self.detection_label.config(text=f"Plaques détectées: {detection_count}")
            self.status_label.config(text=f"Traitement: {progress:.1f}%")
        elif msg_type == "complete":
            if len(message) == 2:
                detection_count = message[1]
                self.processing = False
                self.status_label.config(text="Terminé")
                self.detection_label.config(text=f"Plaques détectées: {detection_count}")
                messagebox.showinfo("Succès", f"{detection_count} plaques détectées.")
            elif len(message) == 3:
                detection_count, plate_text = message[1], message[2]
                self.processing = False
                self.status_label.config(text="Terminé")
                if detection_count > 0:
                    self.detection_label.config(text=f"Plaques détectées: {detection_count}")
                    messagebox.showinfo("Succès", f"Plaque: {plate_text}")
                else:
                    messagebox.showinfo("Information", plate_text)
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
            self.status_label.config(text="Surveillance en cours...")
            self.detection_label.config(text="Plaques détectées: 0")
            self.root.update()

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
        self.status_label.config(text="Surveillance terminée")
        self.camera_btn.config(text="Démarrer Surveillance Caméra")

    def import_image(self):
        if self.processing:
            messagebox.showwarning("Attention", "Un traitement est déjà en cours.")
            return

        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Fichiers image", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"), ("Tous les fichiers", "*.*")]
        )

        if file_path:
            self.processing = True
            self.status_label.config(text=f"Traitement: {os.path.basename(file_path)}")
            self.progress_bar['value'] = 0
            self.root.update()

            thread = threading.Thread(target=self.process_image_in_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
        else:
            self.status_label.config(text="Aucune image sélectionnée")

    def process_image_in_thread(self, file_path):
        self.detector.process_image_file(file_path, self.progress_queue)

    def import_image_manual(self):
        """Import d'image avec sélection manuelle"""
        if self.processing:
            messagebox.showwarning("Attention", "Un traitement est déjà en cours.")
            return

        file_path = filedialog.askopenfilename(
            title="Sélectionner une image pour découpage manuel",
            filetypes=[("Fichiers image", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"), ("Tous les fichiers", "*.*")]
        )

        if file_path:
            self.processing = True
            self.status_label.config(text=f"Mode manuel: {os.path.basename(file_path)}")
            self.root.update()

            thread = threading.Thread(target=self.process_image_manual_in_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
        else:
            self.status_label.config(text="Aucune image sélectionnée")

    def process_image_manual_in_thread(self, file_path):
        result = self.detector.try_manual_cropping(file_path)
        if result:
            plate_text, saved_path = result
            self.db_manager.save_to_database(plate_text, saved_path, f"image: Manuel - {os.path.basename(file_path)}")
            self.progress_queue.put(("complete", 1, plate_text))
        else:
            self.progress_queue.put(("complete", 0, "Aucune plaque détectée en mode manuel"))
        self.processing = False

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
            self.status_label.config(text=f"Traitement vidéo: {os.path.basename(file_path)}")
            self.progress_bar['value'] = 0
            self.detection_label.config(text="Plaques détectées: 0")
            self.root.update()

            thread = threading.Thread(target=self.process_video_in_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
        else:
            self.status_label.config(text="Aucune vidéo sélectionnée")

    def process_video_in_thread(self, file_path):
        self.detector.process_video(file_path, self.progress_queue)

    def show_results(self):
        results_window = tk.Toplevel(self.root)
        results_window.title("Plaques Détectées")
        results_window.geometry("800x500")

        try:
            rows = self.db_manager.get_all_plates()

            frame = tk.Frame(results_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            text_widget = tk.Text(frame, yscrollcommand=scrollbar.set, wrap=tk.WORD, width=90, height=20)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar.config(command=text_widget.yview)

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
                text_widget.insert(tk.END, "Aucune plaque détectée")

            text_widget.config(state=tk.DISABLED)

            export_btn = tk.Button(results_window, text="Exporter les Résultats",
                                   command=self.export_results)
            export_btn.pack(pady=10)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les résultats: {e}")

    def export_results(self):
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
