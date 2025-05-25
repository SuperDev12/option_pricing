import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Black-Scholes calculation functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calculate option Greeks"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    call_theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    put_theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    call_rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return {
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'call_theta': call_theta, 'put_theta': put_theta,
        'vega': vega, 'call_rho': call_rho, 'put_rho': put_rho
    }

# Streamlit app
st.set_page_config(page_title="Black-Scholes Option Pricing Calculator", layout="wide")

st.title("🎯 Black-Scholes Option Pricing Calculator")
st.markdown("Calculate European option prices and Greeks using the Black-Scholes model")

# Sidebar for inputs
st.sidebar.header("📊 Option Parameters")

with st.sidebar.form("option_inputs"):
    st.subheader("Market Parameters")
    S = st.number_input("Current Asset Price ($)", min_value=0.01, value=100.0, step=0.01)
    K = st.number_input("Strike Price ($)", min_value=0.01, value=105.0, step=0.01)
    
    st.subheader("Time and Rates")
    T = st.number_input("Time to Maturity (years)", min_value=0.001, value=1.0, step=0.01)
    r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
    
    st.subheader("Volatility")
    sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=0.1) / 100
    
    submitted = st.form_submit_button("Calculate Options")

if submitted:
    # Calculate option prices
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    greeks = calculate_greeks(S, K, T, r, sigma)
    
    # Main results display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Call Option")
        st.metric("Call Price", f"${call_price:.4f}")
        st.metric("Delta", f"{greeks['call_delta']:.4f}")
        st.metric("Theta", f"{greeks['call_theta']:.4f}")
        st.metric("Rho", f"{greeks['call_rho']:.4f}")
    
    with col2:
        st.subheader("📉 Put Option")
        st.metric("Put Price", f"${put_price:.4f}")
        st.metric("Delta", f"{greeks['put_delta']:.4f}")
        st.metric("Theta", f"{greeks['put_theta']:.4f}")
        st.metric("Rho", f"{greeks['put_rho']:.4f}")
    
    # Shared Greeks
    st.subheader("🔄 Shared Greeks")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Gamma", f"{greeks['gamma']:.6f}")
    with col4:
        st.metric("Vega", f"{greeks['vega']:.4f}")
    
    # Sensitivity Analysis
    st.subheader("📊 Sensitivity Analysis")
    
    # Price sensitivity to underlying asset price
    price_range = np.linspace(S * 0.7, S * 1.3, 50)
    call_prices = [black_scholes_call(price, K, T, r, sigma) for price in price_range]
    put_prices = [black_scholes_put(price, K, T, r, sigma) for price in price_range]
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=price_range, y=call_prices, name='Call Option', line=dict(color='green')))
    fig_price.add_trace(go.Scatter(x=price_range, y=put_prices, name='Put Option', line=dict(color='red')))
    fig_price.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
    fig_price.update_layout(title="Option Prices vs Underlying Asset Price", 
                           xaxis_title="Asset Price ($)", yaxis_title="Option Price ($)")
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Volatility sensitivity
    vol_range = np.linspace(0.1, 0.5, 30)
    call_vol_prices = [black_scholes_call(S, K, T, r, vol) for vol in vol_range]
    put_vol_prices = [black_scholes_put(S, K, T, r, vol) for vol in vol_range]
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vol_range*100, y=call_vol_prices, name='Call Option', line=dict(color='green')))
    fig_vol.add_trace(go.Scatter(x=vol_range*100, y=put_vol_prices, name='Put Option', line=dict(color='red')))
    fig_vol.add_vline(x=sigma*100, line_dash="dash", annotation_text="Current Volatility")
    fig_vol.update_layout(title="Option Prices vs Volatility", 
                         xaxis_title="Volatility (%)", yaxis_title="Option Price ($)")
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Summary table
    st.subheader("📋 Summary Table")
    summary_data = {
        'Metric': ['Call Price', 'Put Price', 'Call Delta', 'Put Delta', 'Gamma', 'Call Theta', 'Put Theta', 'Vega', 'Call Rho', 'Put Rho'],
        'Value': [f"${call_price:.4f}", f"${put_price:.4f}", f"{greeks['call_delta']:.4f}", 
                 f"{greeks['put_delta']:.4f}", f"{greeks['gamma']:.6f}", f"{greeks['call_theta']:.4f}",
                 f"{greeks['put_theta']:.4f}", f"{greeks['vega']:.4f}", f"{greeks['call_rho']:.4f}", f"{greeks['put_rho']:.4f}"]
    }
    st.table(pd.DataFrame(summary_data))

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
This calculator uses the Black-Scholes model to price European options.

**Parameters:**
- **S**: Current asset price
- **K**: Strike price
- **T**: Time to maturity (years)
- **r**: Risk-free rate (decimal)
- **σ**: Volatility (decimal)

**Greeks:**
- **Delta**: Price sensitivity to underlying
- **Gamma**: Delta sensitivity to underlying
- **Theta**: Time decay
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity
""")
