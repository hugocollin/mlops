import streamlit as st
import requests

st.title("My beautiful App")
button_clicked = st.button("Click me")

if button_clicked:
    st.write("It worked")
    st.balloons()

st.write("Prédiction de l'Iris")

sepal_length = st.number_input("Longueur du sépale", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Largeur du sépale", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Longueur du pétale", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Largeur du pétale", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Prédire"):
    response = requests.post("http://server:8000/predict", json={
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    })
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.write(f"La classe prédite est : {prediction}")
    else:
        st.write("Erreur lors de la prédiction")