import streamlit as st
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from options_builder import BlackScholes

st.set_page_config(layout="wide")
st.title("Stock Option Pricer")


#Inputs widgets sidebar
with st.sidebar:
    st.title("Option Parameters")

    time = st.number_input("Time to maturity (in days)", value="min", min_value=1.0,
                                   placeholder="Type a number", step=1.0, format="%.3f")
    spot_price = st.number_input("Spot Price", value="min", min_value=0.1,
                                 placeholder="Type a number", step=0.5)
    strike_price = st.number_input("Strike Price", value="min", min_value=0.1,
                                   placeholder="Type a number", step=0.5)
    volatility = st.number_input("Volatility %", value="min", min_value=0.01,
                                 placeholder="Type a number", step=0.01)
    r_interest = st.number_input("Risk-Free Rate %", value="min",
                                 placeholder="Type a number", step=0.01)
    q_yield = st.number_input("Dividend Yield %", value="min", min_value=0.0,
                              placeholder="Type a number", step=0.01)

table, sensitivity = st.columns([1,2], gap='large')

with table:
    #Row 1: Table with option price and greeks
    time_t = time/252
    sigma = volatility/100
    rfrate = r_interest/100
    qyield = q_yield/100

    call = BlackScholes(spot_price, strike_price, time_t, rfrate, sigma, qyield, 'call')
    put = BlackScholes(spot_price, strike_price, time_t, rfrate, sigma, qyield, 'put')

    table_index = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    table_call = [call.price(), call.delta(), call.gamma(), call.vega(), call.theta(), call.rho()]
    table_put = [put.price(), put.delta(), put.gamma(), put.vega(), put.theta(), put.rho()]
    data = {'Call': table_call,
            'Put': table_put}
    
    data_df = pd.DataFrame(data, table_index)
    st.subheader("Option data")
    st.dataframe(data_df)



greeks_sensi = st.container()
#Row 2 with radio and graphs for greek

with sensitivity:
    #Radio
    st.subheader("Option Greeks Sensitivity")
    col1, col2 = st.columns(2)

    with col1: 
        option_type = st.radio("Select Option Type:", ['Call', 'Put'], horizontal=True)
    with col2:
        greek = st.radio("Select Greek:", ["Delta", "Gamma", "Vega", "Theta", "Rho"], horizontal=True)

    #Graphs for option greeks
    st.subheader(f"Option price sensitivity to {greek}")
    fig = call.greek_sensitivity(greek.lower()) if option_type == "Call" else put.greek_sensitivity(greek.lower())
    st.line_chart(fig, x_label='Spot Price', y_label={greek})



# Row 3 with sensitivity test
sliders, heatmap = st.columns([1,2], gap="medium")

#Sliders to shock option parameters for the heatmap
with sliders:
    st.subheader("Parameter sensitivity")
    op_type = st.radio("Select Option Type:", ['Call', 'Put'], horizontal=True, key=2.0)

    #Map inputs for the Pricing function
    input_mapping = {'Time': 'T',
                     'Spot Price': 'S',
                     'Strike Price': 'K',
                     'Volatility': 'sigma',
                     'Risk-Free Rate': 'r',
                     'Dividend Yield': 'q'}

    selected_params = st.multiselect("Select two parameters to vary:", 
                                     ["Time", "Spot Price", "Strike Price", "Volatility", "Risk-Free Rate", "Dividend Yield"], 
                                     default=["Time", "Spot Price"],
                                     max_selections=2)

    input_vals = {"Time": time_t,
                  "Spot Price": spot_price,
                  "Strike Price": strike_price,
                  "Volatility": sigma,
                  "Risk-Free Rate": rfrate,
                  "Dividend Yield": qyield}
    
    mapping_slider = {"Time": (-99.0, 99.0),
                      "Spot Price": (-99.0, 99.0),
                      "Strike Price": (-99.0, 99.0),
                      "Volatility": (),
                      "Risk-Free Rate": (),
                      "Dividend Yield": ()}
    
    values = {}
    for param in selected_params:
        values[param] = st.slider(param, value=(-99.0, 99.0))
    
with heatmap:
    if len(selected_params)>=2:
        range1_factor = tuple((x/100)+1 for x in values[selected_params[0]])
        range2_factor = tuple((x/100)+1 for x in values[selected_params[1]])

        range1 = tuple(x*input_vals[selected_params[0]] for x in range1_factor)
        range2 = tuple(x*input_vals[selected_params[1]] for x in range2_factor)

        fig, ax = plt.subplots()
        call.heatmap(param1=input_mapping[selected_params[0]], range1=range1, param2=input_mapping[selected_params[1]], 
        range2=range2) if op_type == "Call" else \
        put.heatmap(param1=input_mapping[selected_params[0]], range1=range1, param2=input_mapping[selected_params[1]], range2=range2)
        st.write(fig)
   
