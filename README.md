# README : MLOPS

## Description


## Table des Matières
- [Description](#description)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Description de l'application](#description-de-lapplication)
- [Modèles utilisés](#modèles-utilisés)
- [Contribution](#contribution)
- [Auteur](#auteur)  


## Installation
Pour installer ce projet, suivez ces étapes :
1) Assurez vous d'avoir Docker Desktop installé sur votre machine (https://www.docker.com/products/docker-desktop/) 

2) Clonez le dépôt GitHub sur votre machine :
```bash
git clone https://github.com/hugocollin/mlops
```

3) Allez à la racine du projet dans votre terminal puis exécutez la commande suivante :
```bash
docker-compose up --build
```

- Container Mongo qui stocke les prédictions utilisateur
- API renvoie les images de la prédiction de fleur
- Une page supplémentaire pour afficher les métriques d'apprentissage du modèle (AUC, PR...)
- Train plusieurs modèles (SVM, Decision Tree, XGBoost....) et choisir un modèle pour la prédiction avec une selectbox
- Uploader fichier CSV pour faire de la prédiction par lots
- Exporter sur huggingface spaces / streamlit cloud
- train.py à exécuter dans un container Docker et stock en local via un volume
- Déployer sur Kubernetes avec Minikube

## Utilisation


## Description de l'application 


## Modèles utilisés


## Contribution
Les contributions sont les bienvenues ! Pour contribuer :
- Forkez le projet.
- Créez votre branche de fonctionnalité (```git checkout -b feature/AmazingFeature```).
- Commitez vos changements (```git commit -m 'Add some AmazingFeature'```).
- Poussez à la branche (```git push origin feature/AmazingFeature```).
- Ouvrez une Pull Request.  


## Auteur
Ce projet a été développé par : Hugo COLLIN