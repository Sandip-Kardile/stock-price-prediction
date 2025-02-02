# 📈 Stock Price Prediction using LSTMs & Flask

🚀 **A deep learning-powered stock price prediction web app using LSTM, Yahoo Finance API, and Flask.**  

## 🔥 Project Overview  
This project predicts future stock prices using **Long Short-Term Memory (LSTM) neural networks**.  
The model is trained on **historical stock data** from Yahoo Finance and deployed using **Flask**.  

---

## 📊 Features  
✅ Fetches **real-time & historical stock data** using `yfinance`  
✅ Preprocesses data and applies **MinMax Scaling**  
✅ Trains an **LSTM-based deep learning model**  
✅ Saves & loads models automatically for reuse  
✅ Web app for **predicting future stock prices**  
✅ **Live data visualization**  
✅ Built with **Flask API & REST endpoints**  

---

## 🛠️ Tech Stack  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)  
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  
![yFinance](https://img.shields.io/badge/Yahoo%20Finance-003366?style=for-the-badge)  
![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)  
![PySpark](https://img.shields.io/badge/PySpark-F7B500?style=for-the-badge&logo=apache-spark&logoColor=white)  
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)  
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=power-bi&logoColor=white)  

---

## 📂 Project Structure  
📦 stock-price-prediction
│-- 📁 models/ # Saved LSTM models
│-- 📁 templates/ # HTML files for Flask web app
│-- 📄 app.py # Flask API for predictions
│-- 📄 model.py # LSTM model training & prediction
│-- 📄 data_fetcher.py # Fetches stock data using Yahoo Finance
│-- 📄 requirements.txt # Python dependencies
│-- 📄 README.md # Project documentation



---

## ⚡ Setup Instructions  

### 🛠️ **1. Install Dependencies**  
Clone the repository and install required libraries:  
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt


🚀 3. Run the Flask App

📌 API Endpoints

🚀 Future Improvements
🔹 Add GRU/RNN models for comparison
🔹 Use Sentiment Analysis for stock trend insights
🔹 Deploy on AWS Lambda / Databricks for scalability

🤝 Contributing
Want to improve this project? Feel free to fork and submit a PR! 😊
