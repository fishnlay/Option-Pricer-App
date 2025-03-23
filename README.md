# Option-Pricer-App

This module contains a Streamlit app for Option pricing. It allows to price Vanilla European Options and the respective greeks.
Also, greek values are visualised and a sensitivity test on options parameters is available with a heatmap.

As of now (3/23/2025), it contains the following pricing models:
- Black-Scholes

The app is not deployed yet but can be run locally.

With following command on the terminal:
```
streamlit run app.py
```

## Interface:

- A sidebar for the option parameter.
- A table containing Call/Put prices and greeks
- A lineplot for greek sensitivity to spot

- A heatmap to visualise parameter sensitivity -> 2 parameters can be viewed

![image](https://github.com/user-attachments/assets/610c07f5-4056-408a-b14a-ae6bb53fb4dd)

- A position builder to simulate a portfolio of options
    - Positions can be added (selecting different maturities and strikes as well as the position (long/short) and the     
      quantity.
- A lineplot to visualise the portfolio given different spot price. The portfolio now and at expiry are displayed.
- Parameter shift (sliders) for time, volatility and rates.

![image](https://github.com/user-attachments/assets/16c4c725-3d2f-46be-b15c-53ff477c1d69)
