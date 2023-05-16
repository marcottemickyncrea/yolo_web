# Application de détection d'incendie **YOLO Fire Detection**

## Onglet Analyse Image/Video
Cette onglet permet de lancer l'analyse d'image du web et les vidéos en format MP4 contenu sur l'ordinateur de l'utilisateur.
Dans le cas des images, un tableau récapitulatif indique les différents labels détectés. En cliquant sur une ligne, cela affiche les coordonnées du cadre correspondant.
Dans le cas d'une vidéo, les prédictions apparaissent sur l'image. Une fois la vidéo terminée ou stoppée avant, l'image correspondant au meilleur score est conservée.
Dans le cas où les formats n'est pas pris en compte, un message d'erreur apparait.

## Onglet Analyse Webcam
Cette onglet permet d'analyser les images issus de la webcam. 
En changeant de page, la webcam se stoppe après quelques secondes et conserve l'image avec le meilleur résultat.

## Onglet Archives
Cette onglet permet de voir les différents analyses faites sur l'application, que ce soit des images, des vidéos, ou issus de la webcam. 

### Supprimer l'image
En cliquant sur la lien 'Supprimer l'image', cela ouvre une nouvelle page.
Dans le cas d'une image, celle avec les prédictions et l'orginale sont présentées. Le tableau rassemblant l'ensemble des prognostics est aussi présentés. 
Dans le cas d'une vidéo (mp4 ou webcam), seul l'image (sensée être la plus représentative) avec les prédictions est présente. 
En cliquant sur le bouton 'Supprimer définitivement', toutes les données liées à cette analyse sont supprimées de la base de données.

### Modifier l'image
Cette page permet uniquement de modifier la phrase qui est associé aux prédictions. En cliquant sur "Modifier le texte", cela modifie définitivement le texte.