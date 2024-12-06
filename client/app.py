from sklearn.datasets import load_iris
import streamlit as st
import requests

# Chargement du dataset Iris
iris = load_iris()

# Titre de l'application
st.title("Random Forest Classifier")
st.write("*for the Iris dataset*")

# Création des onglets
home, train, prediction, about = st.tabs(["Home", "Train", "Prediction", "About"])

# Contenu de l'onglet "Home"
with home:
    st.subheader("Home")
    st.write("This web application allows you to train a Random Forest Classifier on the Iris dataset and to make predictions on the species of an iris flower.")

    st.write("The Iris dataset contains 150 samples of iris flowers. Each sample contains the following features:")
    st.write("- Sepal length (cm)")
    st.write("- Sepal width (cm)")
    st.write("- Petal length (cm)")
    st.write("- Petal width (cm)")

    st.write("The target variable is the species of the iris flower. There are three possible species:")
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
                min_samples_split = [int(x) for x in min_samples_split_input.split(",") if x.strip().isdigit()]
                min_samples_leaf = [int(x) for x in min_samples_leaf_input.split(",") if x.strip().isdigit()]

                # Vérification des contraintes côté client (optionnel mais recommandé)
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

                # Préparation de la charge utile avec les paramètres sélectionnés
                payload = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "test_size": test_size,
                    "cv": cv
                }

                # Envoi de la requête POST au serveur
                response = requests.post("http://server:8000/train", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.toast("✅ Model trained successfully !")

                    # Récupération des meilleurs paramètres et de l'accuracy
                    best_params = data['best_params']
                    n_estimators_best = best_params['n_estimators']
                    max_depth_best = best_params['max_depth']
                    min_samples_split_best = best_params['min_samples_split']
                    min_samples_leaf_best = best_params['min_samples_leaf']
                    accuracy = data['test_score']

                    # Affichage des meilleurs paramètres
                    success_message = f"""
                    **Best parameters :**\n
                    - n_estimators : {n_estimators_best}\n
                    - max_depth : {max_depth_best}\n
                    - min_samples_split : {min_samples_split_best}\n
                    - min_samples_leaf : {min_samples_leaf_best}\n
                    """
                    st.success(success_message)
                    st.success(f"**Accuracy : {accuracy:.4f}**")
                else:
                    st.error(f"Error during training: {response.json()['detail']}")
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
    petal_width = st.slider("Leaf width (cm)", min_value=min_petal_width, max_value=max_petal_width, value=1.20, step=0.01)

    # Bouton de lancement de la prédiction
    if st.button("Launch prediction 🚀"):
        try:
            # Envoi de la requête POST au serveur
            response = requests.post("http://server:8000/predict", json={
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            })
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                species = ["Setosa", "Versicolor", "Virginica"]
                st.write(f"The predicted iris species is : {species[prediction]}")
            else:
                st.write("Error while predicting the iris species")
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

# Contenu de l'onglet "About"
with about:
    st.subheader("About")
    st.write("This app was made by me with ❤️")
    st.write("I hope you like it !")
    st.write("Link to the source code : https://github.com/hugocollin/mlops")
    st.write("Link to my website : https://fr.linkedin.com/in/hugocollin23")