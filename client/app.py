from sklearn.datasets import load_iris
import streamlit as st
import requests

# Chargement du dataset Iris
iris = load_iris()

st.title("My beautiful app 🌸")

home, train, prediction, about = st.tabs(["Home", "Train", "Prediction", "About"])

with home:
    st.subheader("Welcome to my app !")
    button_clicked = st.button("🎉 Click me 🎉")

    if button_clicked:
        st.write("It worked !")
        st.balloons()
        st.markdown("""
        <iframe src="https://giphy.com/embed/IwAZ6dvvvaTtdI8SD5" width="480" height="398" style="" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
        """, unsafe_allow_html=True)

with train:
    st.subheader("Train the model")

    if st.button("Train model 🚀"):
        with st.spinner("Training in progress..."):
            try:
                response = requests.post("http://server:8000/train")
                if response.status_code == 200:
                    data = response.json()
                    st.success("Model trained successfully!")
                    st.write(f"**Best Parameters:** {data['best_params']}")
                    st.write(f"**Test Score:** {data['test_score']:.4f}")
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

    if st.button("Predict"):
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