# CryptoVisionX

## Overview
CryptoVisionX is an AI-powered cryptocurrency price prediction and pattern analysis system that utilizes **computer vision, deep learning, and explainable AI (XAI)** to forecast market trends. Unlike traditional numerical-based prediction models, CryptoVisionX leverages **chart image-based pattern recognition** to improve adaptability, usability, and interpretability for traders.

## Features
- **Chart Image-Based Prediction**: Uses deep learning models to analyze cryptocurrency chart patterns.
- **Hybrid Model (CNN + LSTM)**: Combines **Convolutional Neural Networks (CNNs)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for time-series forecasting.
- **Explainable AI (XAI) with Grad-CAM**: Provides heatmap visualizations to explain model predictions and improve trust.
- **Multi-Cryptocurrency Support**: Generalized model trained on various cryptocurrency pairs (e.g., BTCUSDT, ETHUSDT).
- **User-Friendly Interface**: Accepts chart images as input for easy predictions without requiring numerical data.
- **Scalable & Expandable**: Future enhancements include real-time sentiment analysis and extended time-frame predictions.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow & Keras
- **Feature Extraction**: ResNet50 (Pre-trained CNN model)
- **Time-Series Forecasting**: LSTM (Recurrent Neural Network)
- **Explainable AI**: Grad-CAM
- **Data Handling & Visualization**: Pandas, Matplotlib, Seaborn
- **Deployment**: Streamlit, Flask (Future expansion for web-based deployment)

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/shavd3/VisionX.git
   cd VisionX
   ```
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Upload a Cryptocurrency Chart Image**: The system processes and extracts key features.
2. **Prediction & Trend Analysis**: The LSTM model forecasts the future price movement.
3. **Explainability with Grad-CAM**: The system highlights significant areas in the image that influenced the prediction.
4. **Decision Making**: Traders can use the insights to make informed trading decisions.

## Model Architecture
- **Feature Extraction**: ResNet50 extracts visual features from chart images.
- **Dimensionality Reduction**: PCA is applied to refine extracted features.
- **Time-Series Forecasting**: LSTM takes the extracted features and predicts future price trends.
- **Explainability Module**: Grad-CAM generates heatmaps to interpret the model's decisions.

## Future Enhancements
- **Real-Time Sentiment Analysis**: Integrate social media/news data for better market insights.
- **Live API Integration**: Sync with real-time market prices for continuous predictions.
- **Multi-Timeframe Predictions**: Expand the model to handle different trading timeframes.
- **Improved Model Accuracy**: Explore Transformer-based architectures like Informer.

## Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions, reach out to **shavin2001d@gmail.com** or connect via [LinkedIn](https://www.linkedin.com/in/shavin-fernando-d3).

