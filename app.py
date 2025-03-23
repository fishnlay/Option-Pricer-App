import streamlit as st
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from options_builder import BlackScholes, PositionBuilder

st.set_page_config(layout="wide")
st.title("Stock Option Pricer")


#SIDEBAR
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

#ROW 1: TABLE AND SENSITIVITY OF GREEKS
table, sensitivity = st.columns([1,2.2], gap='large')

with table:
    time_t = time/252
    sigma = volatility/100
    rfrate = r_interest/100
    qyield = q_yield/100

    call = BlackScholes(spot_price, strike_price, time_t, rfrate, sigma, qyield, 'call', 'long')
    put = BlackScholes(spot_price, strike_price, time_t, rfrate, sigma, qyield, 'put', 'long')

    table_index = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    table_call = [call.price(), call.delta(), call.gamma(), call.vega(), call.theta(), call.rho()]
    table_put = [put.price(), put.delta(), put.gamma(), put.vega(), put.theta(), put.rho()]
    data = {'Call': table_call,
            'Put': table_put}
    
    data_df = pd.DataFrame(data, table_index)
    st.subheader("Option data")
    st.dataframe(data_df)


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


# ROW 2: HEATMAP AND PARAMETER SELECTION
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
                                     default=["Volatility", "Spot Price"],
                                     max_selections=2)

    input_vals = {"Time": time_t,
                  "Spot Price": spot_price,
                  "Strike Price": strike_price,
                  "Volatility": sigma,
                  "Risk-Free Rate": rfrate,
                  "Dividend Yield": qyield}
    
    mapping_slider = {"Time": (-20.0, 20.0),
                      "Spot Price": (-80.0, 80.0),
                      "Strike Price": (-80.0, 80.0),
                      "Volatility": (-20.0, 20.0),
                      "Risk-Free Rate": (-20.0, 20.0),
                      "Dividend Yield": (-20.0, 20.0)}
    
    values = {}
    for param in selected_params:
        values[param] = st.slider(param, value=mapping_slider[param])
    
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
   
# ROW 3: POSITION BUILDER

st.title("Position builder")

option_selection, visualisation = st.columns([1,3.5], gap='large')

with option_selection:
    st.subheader("Add positions")

    option_type_pos = st.radio("Select Option Type:", ['Call', 'Put'], horizontal=True, key='radio_pos')

    time_pos = st.number_input("Time to maturity (in days)", value="min", min_value=1.0,
                                   placeholder="Type a number", step=1.0, format="%.3f", key='time_pos')
    
    strike_price_pos = st.number_input("Strike Price", value="min", min_value=0.1,
                                   placeholder="Type a number", step=0.5, key='strike_pos')
    
    quantity = st.number_input("Quantity", value="min", min_value=1.0,
                                   placeholder="Type a quantity", step=1.0, key='q_pos')

    left, right = st.columns(2, gap='small')
    with left:
        long = st.button("Long", type='primary')
    with right:
        short = st.button("Short", type='secondary')
    

    #Add the addimulated positions to the portfolio
    if 'pos' not in st.session_state:
        st.session_state.pos = PositionBuilder()
    
    if long or short:
        option_pos = 'long' if long else 'short'
        option = BlackScholes(spot_price, strike_price_pos, time_pos/252, rfrate, sigma, qyield, 
                              option_type_pos, option_pos, quantity)
        st.session_state.pos.add_option(option)


with visualisation:
    time_shift, vol_shift, rate_shift = st.columns(3)

    with time_shift:
        new_T = st.slider("Time Slider (days)", min_value=0.0, max_value=time_pos, value=0.0)

    with vol_shift:
        new_vol = st.slider("Volatility Shift (%)", min_value=-70.0, max_value=200.0, value=0.0)

    with rate_shift:
        new_r = st.slider("Annualised Rate Shift (%)", min_value=-20.0, max_value=20.0, value=0.0)

    st.session_state.pos.shift_parameter('T', new_T)
    st.session_state.pos.shift_parameter('sigma', new_vol)
    st.session_state.pos.shift_parameter('r', new_r)

    #Plot the position 
    fig = st.session_state.pos.positions(spot_price)
    st.line_chart(fig, x_label='Spot Price', y_label='PnL', height=500)


# ROW 4: Table with simulated portfolio
data_pos = pd.DataFrame(st.session_state.pos.pos_table())
data_pos[['Delta', 'Theta', 'Vega', 'Rho']]= data_pos[['Delta', 'Theta', 'Vega', 'Rho']].apply(lambda x: x.map('{:.3f}'.format))
data_pos[['Price', 'Strike']] = data_pos[['Price', 'Strike']].apply(lambda x: x.map('${:.2f}'.format))
data_pos[['Expiry Date']] = data_pos[['Expiry Date']].apply(lambda x: x.map('{:.0f} days'.format))
data_pos[['Gamma']] = data_pos[['Gamma']].apply(lambda x: x.map('{:.4f}'.format))

st.subheader("Simulated Portfolio")
st.dataframe(data_pos)

selection, remove, clear = st.columns([1,1,1], vertical_alignment='bottom')
with selection:
    selected_row = st.selectbox("Select a row to remove", data_pos.index)
with remove:
    if st.button('Remove Selected Row'):
        data_pos = data_pos.drop(selected_row)
        data_pos.reset_index(drop=True, inplace=True)
with clear:
    if st.button("Clear List"):
        st.session_state.pos.options.clear()


