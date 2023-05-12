from flask import Flask, request, flash, render_template, redirect, Response, send_file
from pymongo import MongoClient
import json
from bson import json_util

from my_utils import enregistrer_img, suppression_image, verif_format_img, pourcentages_positions_predict, ajout_cadres, phrase_labels
from my_utils import verif_mp4, predict_video_cam

CONNECTION_STRING = "mongodb://root:1234@localhost:27018/"
client = MongoClient(CONNECTION_STRING)

db = client['yolo']
collection = db['labels']
collection_video = db['labels_video']

app = Flask(__name__)
app.secret_key = '0000'

url_video = ''


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    global url_video

    if request.method == 'POST':
        url = request.form.get('img_url')
        if verif_format_img(url):
            try:                
                results = {}
                id = collection.find_one({}, sort=[('_id', -1)])
                if id is not None:              
                    results['id'] = id['id'] + 1
                else: 
                    results['id'] = 0
                results['url_img'] = {}

                enregistrer_img(url)
                
                results['resultat_predict'] = pourcentages_positions_predict()
                url_img_detect = ajout_cadres(results, results['id'])

                
                results['url_img']['url_img_analyse'] = url
                results['url_img']['url_img_detect'] = url_img_detect.replace('static/', '')

                labels = []
                for key, value in results['resultat_predict'].items():
                    labels.append(key)
                labels_join = ",".join(labels)

                results['phrase'] = phrase_labels(labels_join)

                # envoie du dictionnaire dans mongodb
                json_data_predict = json.loads(json.dumps(results, default=json_util.default))
                collection.insert_one(json_data_predict)  

                return render_template('index.html', response = results)
            except:
                flash("Le site qui héberge l'image n'autorise pas son téléchargement !")
                return render_template('index.html')       
        elif verif_mp4(url):
            url_video = url
            return redirect('/video')
        else:
            return redirect('/')

@app.route('/video')
def accueil_video():
    return render_template('video.html')

@app.route('/predict_video')
def predict_video():      
    predictions = predict_video_cam(url_video, collection)
    return Response(predictions,
        mimetype='multipart/x-mixed-replace; boundary=frame')               

stop = ""            
@app.route('/cam', methods=['GET', 'POST'])
def accueil_cam():
    global stop
    if request.method == 'GET':
        return render_template('cam.html')
    elif request.method == 'POST':        
        stop = request.form.get('stop')
        print(stop)
        return redirect('/')

@app.route('/predict_cam')
def predict_cam():     
        return Response(predict_video_cam(0, collection),
                    mimetype='multipart/x-mixed-replace; boundary=frame')             

    
@app.route('/archives', methods=['GET'])
def archives():
    if request.method == 'GET':
        archives = collection.find({}, {'_id':0})
        archives= [doc for doc in archives]

        return render_template('archives.html', archives= archives)
    
@app.route('/supprimer/<int:id>', methods=['GET', 'POST'])
def supprimer_image(id):
    if request.method == 'GET':
        img_search = collection.find_one({'id': id}, {'_id': 0})
        return render_template('supprimer.html', img_search=img_search)
    
    if request.method == 'POST':
        img_delete = collection.find_one({"id": id}, {"resultat_predict" : 0, "_id": 0, 'phrase': 0})
  
        suppression_image('static/' + img_delete['url_img']['url_img_detect'])
          
        collection.delete_one({"id": id})

        flash("L'image a été supprimée !")

        return redirect('/')

@app.route('/modifier/<int:id>', methods=['GET', 'POST'])
def modifier_image(id):
    if request.method == 'GET':
        return render_template('modifier.html', id=id)
    
    if request.method == 'POST':
        img_texte = request.form.get('img_texte')
        collection.update_one({"id": id}, {"$set":{'phrase': f'{img_texte} (mod)'}})

        flash("Le commentaire de l'image a été modifié !")

        return redirect('/')

if __name__ == '__main__':
    app.run()

# container docker: docker-mongod2
# docker exec -it mongo_db bash
# mongosh "mongodb://root:1234@localhost:27017"