# Battery-Temperature-Estimation

## Battery Temperature Forecasting Using Time Series Model

As battery technology has advanced, the need for accurate and reliable methods to monitor battery health and performance has become crucial. Temperature, in particular, plays a significant role in determining battery performance. Monitoring battery temperature is vital for maintaining optimal operational conditions, preventing thermal runaway, and extending battery life. However, traditional methods for temperature monitoring limit real-time tracking and predictive capabilities. 

Recent advances in time series forecasting have opened up new opportunities for accurately predicting battery temperature trends and detecting anomalies. Lithium-ion batteries, widely used in electronic devices, have demonstrated considerable improvements in their electrical characteristics. Monitoring how temperature changes during battery operations such as charging, discharging, or impedance activities can help detect any anomalies and reveal patterns over time. 

By capturing battery temperature data at evenly spaced intervals, it becomes possible to apply time series forecasting techniques to predict future temperature values. This research focuses on understanding temperature variations in discrete lithium-ion battery cells over time, utilizing a dataset provided by a manufacturer. The dataset contains time series data from 34 unique battery types, each tested in packs of four under identical conditions, with key attributes such as temperature, current, voltage, and time being recorded at regular intervals.

Due to the complexity of the relationship between temperature and other variables, both statistical techniques and machine learning methods have been explored, including ARIMA-based statistical methods, as well as univariate and multivariate LSTM approaches. These approaches have shown promising results in recent projects.

The goal of this study is to develop a generalized forecasting model for each battery type. Each battery type has multiple associated files, containing temperature and other relevant measurements at different points in time. We apply forecasting models to these files and evaluate their performance using the Root Mean Square Error (RMSE) metric. The model with the lowest RMSE for most files is deemed the best fit for that battery type.

Overall, while the ARIMA model remains a robust method for short-term battery temperature prediction, it is important to note that it can handle various patterns such as linear or nonlinear trends, and stationary or non-stationary time series. Additionally, ARIMA requires fewer assumptions and variables, making it simpler to implement and interpret. The performance of both ARIMA and LSTM models is assessed using RMSE as the loss function.

## Research Paper Link: https://link.springer.com/chapter/10.1007/978-981-97-3556-3_5
