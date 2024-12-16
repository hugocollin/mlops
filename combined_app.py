import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import streamlit as st
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement du dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Chemin absolu pour model.pkl
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Fonction d'entraînement du modèle
def train_model(params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=42)
    clf = RandomForestClassifier()
    grid = GridSearchCV(
        estimator=clf,
        param_grid={
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'min_samples_leaf': params['min_samples_leaf']
        },
        cv=params['cv']
    )
    grid.fit(X_train, y_train)
    joblib.dump(grid.best_estimator_, model_path)
    accuracy = grid.score(X_test, y_test)
    return grid.best_params_, accuracy

# Fonction de prédiction
def predict_species(features):
    if not os.path.exists(model_path):
        return None, "Modèle non entraîné."
    model = joblib.load(model_path)
    prediction = model.predict([features])[0]
    species = ["Setosa", "Versicolor", "Virginica"]
    return species[prediction], None

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌸"
)

# Titre de l'application
st.title("Random Forest Classifier")
st.write("*for the Iris dataset*")

# Création des onglets
home, train, prediction, about = st.tabs(["Home", "Train", "Prediction", "About"])

# Contenu de l'onglet "Home"
with home:
    st.subheader("Home")
    st.write("This web application allows you to train a Random Forest Classifier on the Iris dataset and to make predictions on the species of an iris flower.")

    st.write("The Iris dataset contains 150 samples of iris flowers. Each sample contains the following features :")
    st.write("- Sepal length (cm)")
    st.write("- Sepal width (cm)")
    st.write("- Petal length (cm)")
    st.write("- Petal width (cm)")

    st.write("The target variable is the species of the iris flower. There are three possible species :")
    st.write("- Setosa")
    st.write("- Versicolor")
    st.write("- Virginica")

    st.write("The goal is to train a Random Forest Classifier on this dataset and to predict the species of a flower based on its features.")

    st.write("To train the model, go to the 'Train' tab and click on the 'Launch training' button. Once the model is trained, you can make predictions on the 'Prediction' tab by entering the features of the flower and clicking on the 'Launch prediction' button.")

    st.write("Enjoy ! 🔥")

# Contenu de l'onglet "Train"
with train:
    st.subheader("Train the model")

    # Paramètres du modèle
    n_estimators_input = st.text_input("Number of estimators", "100,200,300")
    max_depth_input = st.text_input("Maximum depth of the trees", "10,20,30")
    include_none = st.checkbox("Include 'None' value for max depth", value=False)
    min_samples_split_input = st.text_input("Minimum number of samples required to split a node", "2,5,10")
    min_samples_leaf_input = st.text_input("Minimum number of samples required at a leaf node", "1,2,4")
    test_size = st.slider("Test size", min_value=0.1, max_value=0.9, value=0.3, step=0.01)
    cv = st.slider("Number of folds for cross-validation", min_value=2, max_value=10, value=5, step=1)

    # Bouton de lancement de l'entraînement
    if st.button("Launch training 🚀"):
        with st.spinner("Training in progress..."):
            try:
                # Conversion des entrées en listes de nombres
                n_estimators = [int(x) for x in n_estimators_input.split(",") if x.strip().isdigit()]
                max_depth = [int(x) for x in max_depth_input.split(",") if x.strip().isdigit()]
                if include_none:
                    max_depth.append(None)
                min_samples_split = [int(x) for x in min_samples_split_input.split(",") if x.strip().isdigit()]
                min_samples_leaf = [int(x) for x in min_samples_leaf_input.split(",") if x.strip().isdigit()]

                # Vérification des contraintes côté client
                if not n_estimators:
                    st.error("Please enter at least one valid value for the number of estimators.")
                    st.stop()
                if not max_depth:
                    st.error("Please enter at least one valid value for the maximum depth of the trees.")
                    st.stop()
                if not min_samples_split:
                    st.error("Please enter at least one valid value for the minimum number of samples required to split a node.")
                    st.stop()
                if not min_samples_leaf:
                    st.error("Please enter at least one valid value for the minimum number of samples required at a leaf node.")
                    st.stop()

                # Préparation des paramètres
                params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "test_size": test_size,
                    "cv": cv
                }

                # Entraînement du modèle
                best_params, accuracy = train_model(params)
                st.success("✅ Model trained successfully !")

                # Affichage des meilleurs paramètres
                success_message = f"""
                **Best parameters :**\n
                - n_estimators : {best_params['n_estimators']}\n
                - max_depth : {best_params['max_depth']}\n
                - min_samples_split : {best_params['min_samples_split']}\n
                - min_samples_leaf : {best_params['min_samples_leaf']}\n
                """
                st.success(success_message)
                st.success(f"**Accuracy : {accuracy:.4f}**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Contenu de l'onglet "Prediction"
with prediction:
    st.subheader("Iris species prediction")

    # Récuprération des valeurs min et max pour chaque feature
    min_sepal_length = min(iris.data[:, 0])
    max_sepal_length = max(iris.data[:, 0])
    min_sepal_width = min(iris.data[:, 1])
    max_sepal_width = max(iris.data[:, 1])
    min_petal_length = min(iris.data[:, 2])
    max_petal_length = max(iris.data[:, 2])
    min_petal_width = min(iris.data[:, 3])
    max_petal_width = max(iris.data[:, 3])

    # Paramètres des caractéristiques de la fleur
    sepal_length = st.slider("Sepal length (cm)", min_value=min_sepal_length, max_value=max_sepal_length, value=5.84, step=0.01)
    sepal_width = st.slider("Sepal width (cm)", min_value=min_sepal_width, max_value=max_sepal_width, value=3.05, step=0.01)
    petal_length = st.slider("Petal length (cm)", min_value=min_petal_length, max_value=max_petal_length, value=3.75, step=0.01)
    petal_width = st.slider("Petal width (cm)", min_value=min_petal_width, max_value=max_petal_width, value=1.20, step=0.01)

    # Bouton de lancement de la prédiction
    if st.button("Launch prediction 🚀"):
        with st.spinner("Predicting..."):
            species, error = predict_species([sepal_length, sepal_width, petal_length, petal_width])
            if error:
                st.error(error)
            else:
                st.success(f"The predicted iris species is : **{species}**")

                if species == "Setosa":
                    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg"
                elif species == "Versicolor":
                    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Iris_versicolor_1.jpg/1280px-Iris_versicolor_1.jpg"
                else:
                    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1200px-Iris_virginica_2.jpg"
                st.image(image_url, caption=f"Iris {species}", use_container_width=True)

# Contenu de l'onglet "About"
with about:
    st.subheader("About")
    st.write("This app was made by me with ❤️")
    st.write("I hope you like it !")
    st.write("Link to the source code : https://github.com/hugocollin/mlops")
    st.write("Link to my website : https://fr.linkedin.com/in/hugocollin23")