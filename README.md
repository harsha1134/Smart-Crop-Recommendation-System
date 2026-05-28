# Smart Crop Recommendation System

A machine learning-based web application that recommends suitable crops based on soil and environmental conditions. The system uses a RandomForestClassifier trained on agricultural datasets to predict the most appropriate crop for cultivation.

## Features

* Crop prediction using Machine Learning
* Real-time prediction through web interface
* User-friendly Flask application
* Soil and environmental parameter analysis
* Data preprocessing and feature scaling

## Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML5
* CSS3

## Machine Learning Model

* RandomForestClassifier
* Feature Scaling using:

  * MinMaxScaler
  * StandardScaler

## Input Parameters

The model predicts crops based on:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

## Project Structure

```text
Smart-Crop-Recommendation-System/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
├── static/
├── templates/
├── dataset/
└── notebooks/
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/harsha1134/Smart-Crop-Recommendation-System.git
```

2. Navigate to project folder

```bash
cd Smart-Crop-Recommendation-System
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the Flask application

```bash
python app.py
```

## Future Improvements

* Deep Learning-based prediction models
* Weather API integration
* Mobile application support
* Multi-language support
* Crop yield estimation

## Author

Harshavardhan Pinnika
