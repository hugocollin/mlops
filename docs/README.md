# README : MLOps

## Table of contents
- [Description](#description)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Authors](#authors)

## Description

This project is an MLOps application designed to streamline the training, evaluation, and deployment of a Random Forest Classifier for predicting Iris species. It integrates a robust backend built with FastAPI and a user-friendly frontend developed using Streamlit, all orchestrated within a Dockerized environment for seamless deployment and scalability.

### Key features

- **Interactive model training** : Users can input multiple hyperparameters, including `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. The application supports grid search with cross-validation to identify the optimal model parameters. Additionally, users have the option to include `None` for `max_depth` to allow trees to expand until all leaves are pure.

- **Real-time model evaluation** : After training, the application displays the best hyperparameters and the model's accuracy on the test dataset, providing immediate feedback on model performance.

- **Predictive insights with visuals** : Users can input the characteristics of an Iris flower to predict its species. The prediction result is complemented by an image of the predicted species fetched from a free image API, enhancing the user experience.

- **Database integration** : The backend integrates with MongoDB to manage data storage, allowing for operations such as adding and listing fruits, demonstrating basic database interactions.

- **Dockerized deployment** : Both the client and server applications are containerized using Docker, ensuring consistent environments across different setups and simplifying the deployment process with Docker Compose.

### Project structure

```bash
├── client/
│   ├── app.py              # Streamlit frontend application
│   ├── Dockerfile          # Docker configuration for the client
│   └── requirements.txt    # Python dependencies for the client
├── server/
│   ├── app.py              # FastAPI backend application
│   ├── model.py            # Model training and prediction logic
│   ├── model.pkl           # Serialized trained model
│   ├── Dockerfile          # Docker configuration for the server
│   └── requirements.txt    # Python dependencies for the server
├── docker-compose.yml      # Docker Compose configuration to orchestrate services
└── README.md               # Project documentation
```

## Installation
1. To run this project, you will need to have Docker Desktop installed on your machine (https://www.docker.com/products/docker-desktop/)

2. Clone the repository by running the following command in your terminal :
```bash
git clone https://github.com/hugocollin/mlops
```

3. Navigate to the project directory and run the following command to build the Docker image and start the container :
```bash
docker-compose up --build
```

## Usage

This web application allows you to train a Random Forest Classifier on the Iris dataset and to make predictions on the species of an iris flower.

The Iris dataset contains 150 samples of iris flowers. Each sample contains the following features :

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The target variable is the species of the iris flower. There are three possible species :

- Setosa
- Versicolor
- Virginica

The goal is to train a Random Forest Classifier on this dataset and to predict the species of a flower based on its features.

To train the model, go to the 'Train' tab and click on the 'Launch training' button. Once the model is trained, you can make predictions on the 'Prediction' tab by entering the features of the flower and clicking on the 'Launch prediction' button.

Enjoy ! 🔥

## Contribution

All contributions are welcome. Here's how you can help :

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.  

## Author
This project was developed by COLLIN Hugo a student from the Master 2 SISE program at the University of Lyon 2.