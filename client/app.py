from sklearn.datasets import load_iris
import streamlit as st
import requests

# Chargement du dataset Iris
iris = load_iris()

st.title("Random Forest Classifier")
st.write("*for the Iris dataset*")

home, train, prediction, about = st.tabs(["Home", "Train", "Prediction", "About"])

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

with train:
    st.subheader("Train the model")

    if st.button("Launch training 🚀"):
        with st.spinner("Training in progress..."):
            try:
                response = requests.post("http://server:8000/train")
                if response.status_code == 200:
                    data = response.json()
                    st.toast("✅ Model trained successfully !")
                    # Récupérer les données de réponse
                    data = response.json()

                    # Récupération des meilleurs paramètres
                    best_params = data['best_params']
                    n_estimators = best_params['n_estimators']
                    max_depth = best_params['max_depth']
                    min_samples_split = best_params['min_samples_split']
                    min_samples_leaf = best_params['min_samples_leaf']
                    accuracy = data['test_score']

                    # Affichage des meilleurs paramètres
                    success_message = f"""
                    **Best parameters :**\n
                    - n_estimators : {n_estimators}\n
                    - max_depth : {max_depth}\n
                    - min_samples_split : {min_samples_split}\n
                    - min_samples_leaf : {min_samples_leaf}\n
                    """
                    st.success(success_message)
                    st.success(f"**Accuracy : {accuracy:.4f}**")
                else:
                    st.error(f"Error during training: {response.json()['detail']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

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

    sepal_length = st.slider("Sepal length (cm)", min_value=min_sepal_length, max_value=max_sepal_length, value=5.84, step=0.01)
    sepal_width = st.slider("Sepal width (cm)", min_value=min_sepal_width, max_value=max_sepal_width, value=3.05, step=0.01)
    petal_length = st.slider("Petal length (cm)", min_value=min_petal_length, max_value=max_petal_length, value=3.75, step=0.01)
    petal_width = st.slider("Leaf width (cm)", min_value=min_petal_width, max_value=max_petal_width, value=1.20, step=0.01)

    if st.button("Launch prediction 🚀"):
        response = requests.post("http://server:8000/predict", json={
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        })
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            if prediction == 0:
                prediction = "Setosa"
            elif prediction == 1:
                prediction = "Versicolor"
            elif prediction == 2:
                prediction = "Virginica"
            st.write(f"The predicted iris species is : {prediction}")
        else:
            st.write("Error while predicting the iris species")
    
with about:
    st.subheader("About")
    st.write("This app was made by me with ❤️")
    st.write("I hope you like it !")
    st.write("Link to the source code : https://github.com/hugocollin/mlops")
    st.write("Link to my website : https://fr.linkedin.com/in/hugocollin23")