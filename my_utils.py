import os
import torch
from PIL import Image
import re
import urllib.request
import cv2
import json
from bson import json_util

path = r'static\yolov5\runs\train\results_2\weights\best.pt'

def enregistrer_img(url):
    # Télécharger l'image depuis l'URL
    urllib.request.urlretrieve(url, "static/temp.jpg")

    image = cv2.imread("static/temp.jpg")

    # Enregistrer l'image avec OpenCV
    cv2.imwrite("static/temp.jpg", image)

def suppression_image(url_img):
    os.remove(url_img)

def verif_format_img(url_img):
    """vérifie le format de l'image dans l'url"""
    # Regex pour rechercher des URLs d'images dans la chaîne de caractères
    regex = r"(?i)(https?://\S+\.(?:jpg|jpeg|png|gif))"

    # Recherche des URLs d'images dans la chaîne de caractères
    resultats = bool(re.match(regex, url_img))

    # Affichage des URLs d'images trouvées
    return resultats

def verif_mp4(url):
    """vérifie le format de la vidéo"""
    bool = False
    if url.find('.mp4') != -1:
        bool = True
    return bool


def phrase_labels(phrase):
    """détermine une phrase en fonction des labels détectés dans une phrase"""
    response = ''

    if phrase.find('Feu') != -1:
        if phrase.find('Fumee') != -1:
            if phrase.find('Pompier') != -1:
                response = "Incendie en cours, les pompiers sont déjà à l'oeuvre !"
            else:
                # response = "Incendie en cours, arrête de filmer et appelle les pompier !"
                response = 'INCENDIE en cours !'
        elif phrase.find('Clope') != -1:
            response = "Cloper tue !!!!"
        elif phrase.find('Poele') != -1:
            response = "J'aime la semelle, pas le charbon !"
        elif phrase.find('Bougie') != -1:
            response = "Ambiance romantique détecté, à moins que ..."
        elif phrase.find('Pompier') != -1:
            response = "Incendie en cours, les pompiers sont déjà à l'oeuvre !"
        else:
            # response = "Il y a le feu, oui, mais ça peut manquer de contexte !"
            response = 'Feu détecté !'
    elif phrase.find('Fumee') != -1:
        if phrase.find('Pompier') != -1:
            response = 'Incendie maitrisé, merci aux pompiers'
        else:
            # response = 'Il y a pas de fumée sans feu !'
            response = 'Fumée détectée !'
    elif phrase.find('Clope') != -1:
        response = "Présence d'un serial killer dans cette image !"
    elif phrase.find('Bougie') != -1:
        response = "Il y a des bougie, voilà quoi !"
    elif phrase.find('Poele') != -1:
        response = "Il y a une poele, c'est tout !"
    elif phrase.find('Pompier') != -1:
        response = "Merci aux pompiers !"
    else:
        # response = "Pas de feu, pas de feu !"
        response = 'Rien à signaler !'

    return response


def pourcentages_positions_predict():
    """prédiction sur les images"""
    # Charger le modèle entraîné
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path= path)

    # Ouvrir l'image en tant que fichier image
    img = Image.open('static/temp.jpg')

    # Exécuter la détection
    results = model(img)

    # Récupérer les prédictions
    predictions = results.xyxy[0]

    # Récupérer les noms de classes
    class_names = model.module.names if hasattr(
        model, 'module') else model.names

    # Créer un dictionnaire pour stocker les positions de chaque classe
    positions_par_classe = {}

   # Parcourir les prédictions et stocker les positions dans le dictionnaire
    for pred in predictions:
        x_min, y_min, x_max, y_max, confidence, class_idx = pred
        class_name = class_names[int(class_idx)]
        if class_name not in positions_par_classe:
            positions_par_classe[class_name] = []
        position = {
            'pourcentage': confidence.item(),
            'x_min': x_min.item(),
            'y_min': y_min.item(),
            'x_max': x_max.item(),
            'y_max': y_max.item()
        }
        positions_par_classe[class_name].append(position)
    
    return positions_par_classe


def couleur_label(classe):
    """couleurs des cadres en fonction du label"""
    couleur = ""
    if classe == "Feu":
        couleur = (0, 165, 255)  # orange
    elif classe == "Fummee":
        couleur = (128, 128, 128)  # gris
    elif classe == 'Pompier':
        couleur = (0, 0, 255)  # rouge
    elif classe == 'Poele':
        couleur = (0, 0, 0)  # noir
    elif classe == 'Clope':
        couleur = (128, 0, 128)  # violet
    elif classe == 'Bougie':
        couleur = (203, 192, 255)  # rose
    return couleur

def ajout_cadres(positions, id):
    """ajout de cadre en fonction des coordonnées fournis lors de la prédiction"""
    # Charger l'image
    image = cv2.imread('static/temp.jpg')

    # Parcourir les objets détectés
    for classe, objets in positions['resultat_predict'].items():
        for obj in objets:
            # Extraire les informations
            label = classe
            pourcentage = round(obj['pourcentage'] * 100, 2)
            x_min = int(obj['x_min'])
            y_min = int(obj['y_min'])
            x_max = int(obj['x_max'])
            y_max = int(obj['y_max'])

            # Dessiner le carré sur l'image
            couleur = couleur_label(label)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), couleur, 2)

            # Ajouter un rectangle sous le texte
            (w, h), _ = cv2.getTextSize(
                f"{label}: {pourcentage}%", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(image, (x_min, y_min),
                          (x_min + w, y_min + h), couleur, -1)

            # Ajouter le label et le pourcentage à l'intérieur du rectangle
            texte = f"{label}: {pourcentage}%"
            cv2.putText(image, texte, (x_min, y_min + h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Enregistrer l'image avec les cadres et les informations
    url_predict = f'static/images/predict_{id}.jpg'
    cv2.imwrite(url_predict, image)

    return url_predict

def predict_video_cam(url, collection):
    """prédiction sur les vidéos & cam"""
    response = {}
    labels = {}

    id_last = collection.find_one({}, sort=[('_id', -1)])
    if id_last is not None:
        response['id'] = id_last['id'] + 1
    else:
        response['id'] = 0
    response['url_img'] = {}

    id_video = response['id']

    # Charger votre modèle YOLO entraîné
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path= path)

    # Charger la vidéo
    cap = cv2.VideoCapture(url)

    # Définir le facteur d'accélération
    acceleration_factor = 10

    frame_count = 0  # Compteur de frames
    while True:
        # Lire une frame de la vidéo
        ret, frame = cap.read()

        # Sortir de la boucle si la fin de la vidéo est atteinte
        if not ret:
            break

        # Incrémenter le compteur de frames
        frame_count += 1

        # Accélérer la vidéo en sautant certaines frames
        if frame_count % acceleration_factor != 0:
            continue

        # Effectuer les prédictions avec votre modèle YOLO sur la frame
        results = model(frame)

        # Récupérer les positions des objets détectés
        positions = results.pandas().xyxy[0]

        # Parcourir les objets détectés
        for i, obj in positions.iterrows():
            x_min, y_min, x_max, y_max, conf, cls = obj[:6]
            label = model.names[int(cls)]
            pourcentage = round(conf * 100, 2)

            # Dessiner le carré sur la frame
            couleur = couleur_label(label)
            cv2.rectangle(frame, (int(x_min), int(y_min)),
                          (int(x_max), int(y_max)), couleur, 2)

            # Enregistrer les pourcentages (feu, fumée) les plus élevés
            texte = f"{label}: {pourcentage}%"
            if label in labels:
                if pourcentage > labels[label][0]['pourcentage']:
                    labels[label] = [{'pourcentage': pourcentage,
                                      'x_min': x_min,
                                      'x_max': x_max,
                                      'y_min': y_min,
                                      'y_max': y_max}]
                    if label == "Feu" or label == "Fumee":
                        cv2.imwrite(
                            f'static/images/predict_video_{id_video}.jpg', frame)
            else:
                labels[label] = [{'pourcentage': pourcentage,
                                      'x_min': x_min,
                                      'x_max': x_max,
                                      'y_min': y_min,
                                      'y_max': y_max}]

            # Calculer la largeur et la hauteur du texte
            (w, h), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Dessiner un rectangle pour le texte
            cv2.rectangle(frame, (int(x_min), int(y_min)),
                          (int(x_min) + w, int(y_min) - h - 10), couleur, -1)
            # Ajouter le texte
            cv2.putText(frame, texte, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convertir la frame en format d'image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_buffer = buffer.tobytes()

        response['url_img']['url_img_detect'] = f'images/predict_video_{id_video}.jpg'
        response['resultat_predict'] = labels

        id_last_bis = collection.find_one({}, sort=[('_id', -1)])
        if id_last_bis['id'] == id_video:
            collection.delete_one({'id': id_last_bis['id']})  
            json_data_predict = json.loads(
                json.dumps(response, default=json_util.default))
            collection.insert_one(json_data_predict)
        else:
            cv2.imwrite(f'static/images/predict_video_{id_video}.jpg', frame)
            json_data_predict = json.loads(
                json.dumps(response, default=json_util.default))
            collection.insert_one(json_data_predict)

        # Envoyer la frame à la réponse de l'API
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n\r\n')
    

# ancienne version pour faire les prédictions => prenait beaucoup de temps, pas possible pour  vidéo
# def pourcentages_positions_predict(img_path, id):
#     # Charger le modèle entraîné
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'static\yolov5\runs\train\results_1\weights\best.pt')

#     # Ouvrir l'image en tant que fichier image
#     img = Image.open(img_path)

#     # Exécuter la détection
#     results = model(img)

#     # Récupérer les prédictions
#     predictions = results.xyxy[0]

#     # Récupérer les noms de classes
#     class_names = model.module.names if hasattr(model, 'module') else model.names

#     # Créer un dictionnaire pour stocker les positions de chaque classe
#     positions_par_classe = {}

#    # Parcourir les prédictions et stocker les positions dans le dictionnaire
#     for pred in predictions:
#         x_min, y_min, x_max, y_max, confidence, class_idx = pred
#         class_name = class_names[int(class_idx)]
#         if class_name not in positions_par_classe:
#             positions_par_classe[class_name] = []
#         position = {
#             'pourcentage': confidence.item(),
#             'x_min': x_min.item(),
#             'y_min': y_min.item(),
#             'x_max': x_max.item(),
#             'y_max': y_max.item()
#         }
#         positions_par_classe[class_name].append(position)

#     response = {}
#     response['resultats_predict'] = positions_par_classe
#     response['id'] = id

#     return response


# def run_yolov5_predict(img_path, conf=0.25, iou=0.45):
#     """
#     Run YOLOv5 predict function on an image.
#     :param img_path: path to the input image
#     :param weights_path: path to the trained weights
#     :param conf: confidence threshold (default: 0.25)
#     :param iou: IoU threshold (default: 0.45)
#     :return: a list of detected objects with their labels and bounding boxes
#     """
#     #poids du model utilisé
#     weights_path = r'static\yolov5\runs\train\results_1\weights\best.pt'

#     # commande pour lancer la prédiction
#     command = ["python", "static/yolov5/detect.py", "--source", img_path, "--weights", weights_path, "--conf", str(conf), "--iou", str(iou)]

#     # lance la commande de prédiction
#     output = subprocess.run(command, shell=True, capture_output=True)

#     # création d'un dictionnaire avec les prédictions + url images
#     detections = {}
#     detections['labels'] = {}
#     detections['image'] = {'url_img_analyse': img_path}
#     url_image = ''
#     phrase = ''
#     for line in output.stderr.decode('utf-8').split('\n'):
#         if line.startswith('image'):
#             #ajoute la phrase générée en fonction des labels détectés
#             phrase = phrase_labels(line)
#             detections['phrase'] = phrase
#             if line.find('Feu') != -1:
#                 detections['labels']['feu'] = line[line.index('Feu') - 3: line.index('Feu')]
#             if line.find('Fumee') != -1:
#                 detections['labels']['fumee'] = line[line.index('Fumee') - 3: line.index('Fumee') ]
#             if line.find('Pompier') != -1:
#                 detections['labels']['pompier'] =  line[line.index('Pompier') - 3: line.index('Pompier')]
#             if line.find('Clope') != -1:
#                 detections['labels']['clope'] = line[line.index('Clope') - 3: line.index('Clope')]
#             if line.find('Poele') != -1:
#                 detections['labels']['poele'] =  line[line.index('Poele') - 3: line.index('Poele')]
#             if line.find('Bougie') != -1:
#                 detections['labels']['bougie'] =  line[line.index('Bougie') - 3: line.index('Bougie')]
#         if line.startswith('Downloading'):
#             url_image = f"{line[line.index('to ') + 3: -1].replace('...', '')}"
#         elif line.startswith('Found'):
#            url_image = f"{line[line.index('at ') + 3: -1].replace('...', '')}"
#         if line.startswith('Result'):
#             url_image_detect = f"{line[line.index('yolo'): -5]}/{url_image}"
#             detections['image']['url_img_detect'] = url_image_detect.replace("\\", "/")
#             detections['id'] = f"{line[line.index('exp'): -5]}"
#     print

#     #a ppelle une deuxième fois le modèle pour récupérer les pourcentages et les positions
#     pourcentages_positions = pourcentages_positions_predict(url_image, detections['id'])

#     #suppression de l'image créée par Yolo (ne peut pas être appelé avec flask-extérieur au fichier static-)
#     suppression_image(url_image)

#     return [detections, pourcentages_positions]
