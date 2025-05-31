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

def create_heatmap(K, T, r, min_spot, max_spot, min_vol, max_vol, option_type='call'):
    """Create heatmap data for option prices"""
    spot_prices = np.linspace(min_spot, max_spot, 50)
    volatilities = np.linspace(min_vol, max_vol, 50)
    
    heatmap_data = np.zeros((len(volatilities), len(spot_prices)))
    
    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            if option_type == 'call':
                heatmap_data[i, j] = black_scholes_call(spot, K, T, r, vol)
            else:
                heatmap_data[i, j] = black_scholes_put(spot, K, T, r, vol)
    
    return heatmap_data, spot_prices, volatilities

# Streamlit app configuration
st.set_page_config(page_title="Black-Scholes Option Pricing Calculator", layout="wide")

st.title("🎯 Black-Scholes Option Pricing Calculator")
st.markdown("Calculate European option prices and Greeks using the Black-Scholes model")

# Sidebar for inputs
st.sidebar.header("📊 Option Parameters")

with st.sidebar.form("option_inputs"):
    st.subheader("Market Parameters")
    S = st.number_input("Current Asset Price ($)", 
                       min_value=0.01, 
                       value=100.0, 
                       step=0.01,
                       help="")
    
    K = st.number_input("Strike Price ($)", 
                       min_value=0.01, 
                       value=105.0, 
                       step=0.01,
                       help="")
    
    st.subheader("Time and Rates")
    T = st.number_input("Time to Maturity (years)", 
                       min_value=0.001, 
                       value=1.0, 
                       step=0.01,
                       help="")
    
    r = st.number_input("Risk-Free Rate (%)", 
                       min_value=0.0, 
                       value=5.0, 
                       step=0.1,
                       help="") / 100
    
    st.subheader("Volatility")
    sigma = st.number_input("Volatility (%)", 
                           min_value=0.1, 
                           value=20.0, 
                           step=0.1,
                           help="") / 100
    
    submitted = st.form_submit_button("Calculate Options")

# Custom Market Prices Section
st.sidebar.markdown("---")
st.sidebar.header("💰 Custom Market Prices")

with st.sidebar.form("custom_prices"):
    st.subheader("Market Option Prices")
    use_custom_prices = st.checkbox("Use Custom Market Prices", help="")
    
    custom_call_price = st.number_input("Market Call Price ($)", 
                                       min_value=0.0, 
                                       value=10.0, 
                                       step=0.01,
                                       help="")
    
    custom_put_price = st.number_input("Market Put Price ($)", 
                                      min_value=0.0, 
                                      value=8.0, 
                                      step=0.01,
                                      help="")
    
    custom_submitted = st.form_submit_button("Compare with Market")

# Heatmap controls
st.sidebar.markdown("---")
st.sidebar.header("🔥 Heatmap Controls")

with st.sidebar.form("heatmap_inputs"):
    st.subheader("Spot Price Range")
    min_spot = st.number_input("Minimum Spot Price ($)", 
                              min_value=0.01, 
                              value=80.0, 
                              step=1.0,
                              help="")
    
    max_spot = st.number_input("Maximum Spot Price ($)", 
                              min_value=0.01, 
                              value=120.0, 
                              step=1.0,
                              help="")
    
    st.subheader("Volatility Range")
    min_vol = st.number_input("Minimum Volatility (%)", 
                             min_value=0.1, 
                             value=10.0, 
                             step=0.1,
                             help="") / 100
    
    max_vol = st.number_input("Maximum Volatility (%)", 
                             min_value=0.1, 
                             value=50.0, 
                             step=0.1,
                             help="") / 100
    
    option_type = st.selectbox("Option Type for Heatmap", 
                              ["call", "put"], 
                              help="")
    
    heatmap_submitted = st.form_submit_button("Generate Heatmap")

if submitted:
    # Calculate option prices
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    greeks = calculate_greeks(S, K, T, r, sigma)
    
    # Main results display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Call Option")
        st.metric("Theoretical Call Price", f"${call_price:.4f}")
        st.metric("Delta", f"{greeks['call_delta']:.4f}")
        st.metric("Theta", f"{greeks['call_theta']:.4f}")
        st.metric("Rho", f"{greeks['call_rho']:.4f}")
    
    with col2:
        st.subheader("📉 Put Option")
        st.metric("Theoretical Put Price", f"${put_price:.4f}")
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

# Custom Price Comparison
if custom_submitted and use_custom_prices:
    # Calculate theoretical prices for comparison
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    
    st.subheader("🔍 Market vs Theoretical Price Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Call Option Analysis")
        call_diff = custom_call_price - call_price
        st.metric("Market Call Price", f"${custom_call_price:.4f}")
        st.metric("Theoretical Call Price", f"${call_price:.4f}")
        st.metric("Price Difference", f"${call_diff:.4f}", 
                 delta=f"{(call_diff/call_price)*100:.2f}%" if call_price != 0 else "N/A")
        
        if abs(call_diff) > 0.5:
            if call_diff > 0:
                st.error(f"📉 Call Option appears OVERVALUED by ${call_diff:.2f}")
                st.markdown("**Suggestion:** Consider selling or avoid buying")
            else:
                st.success(f"📈 Call Option appears UNDERVALUED by ${abs(call_diff):.2f}")
                st.markdown("**Suggestion:** Consider buying")
        else:
            st.info("📊 Call Option appears fairly priced")
    
    with col2:
        st.subheader("📉 Put Option Analysis")
        put_diff = custom_put_price - put_price
        st.metric("Market Put Price", f"${custom_put_price:.4f}")
        st.metric("Theoretical Put Price", f"${put_price:.4f}")
        st.metric("Price Difference", f"${put_diff:.4f}", 
                 delta=f"{(put_diff/put_price)*100:.2f}%" if put_price != 0 else "N/A")
        
        if abs(put_diff) > 0.5:
            if put_diff > 0:
                st.error(f"📉 Put Option appears OVERVALUED by ${put_diff:.2f}")
                st.markdown("**Suggestion:** Consider selling or avoid buying")
            else:
                st.success(f"📈 Put Option appears UNDERVALUED by ${abs(put_diff):.2f}")
                st.markdown("**Suggestion:** Consider buying")
        else:
            st.info("📊 Put Option appears fairly priced")
    
    # Comparison table
    st.subheader("📊 Detailed Comparison Table")
    comparison_data = {
        'Option Type': ['Call Option', 'Put Option'],
        'Market Price': [f"${custom_call_price:.4f}", f"${custom_put_price:.4f}"],
        'Theoretical Price': [f"${call_price:.4f}", f"${put_price:.4f}"],
        'Difference ($)': [f"${call_diff:.4f}", f"${put_diff:.4f}"],
        'Difference (%)': [f"{(call_diff/call_price)*100:.2f}%" if call_price != 0 else "N/A",
                          f"{(put_diff/put_price)*100:.2f}%" if put_price != 0 else "N/A"],
        'Assessment': ['Overvalued' if call_diff > 0.5 else 'Undervalued' if call_diff < -0.5 else 'Fair',
                      'Overvalued' if put_diff > 0.5 else 'Undervalued' if put_diff < -0.5 else 'Fair']
    }
    st.table(pd.DataFrame(comparison_data))

if heatmap_submitted:
    # Validate inputs
    if max_spot <= min_spot:
        st.error("Maximum spot price must be greater than minimum spot price")
    elif max_vol <= min_vol:
        st.error("Maximum volatility must be greater than minimum volatility")
    else:
        # Generate heatmap
        st.subheader(f"🔥 {option_type.capitalize()} Option Price Heatmap")
        
        with st.spinner("Generating heatmap..."):
            heatmap_data, spot_prices, volatilities = create_heatmap(
                K, T, r, min_spot, max_spot, min_vol, max_vol, option_type
            )
            
            # Create the heatmap using Plotly
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=np.round(spot_prices, 2),
                y=np.round(volatilities * 100, 1),  # Convert to percentage
                colorscale='RdYlGn',  # Red to Yellow to Green
                hoverongaps=False,
                hovertemplate='Spot Price: $%{x}<br>Volatility: %{y}%<br>Option Price: $%{z:.4f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title=f"{option_type.capitalize()} Option Prices (Strike: ${K}, Time: {T} years, Rate: {r*100:.1f}%)",
                xaxis_title="Spot Price ($)",
                yaxis_title="Volatility (%)",
                width=800,
                height=600
            )
            
            # Add current position marker if within range
            if min_spot <= S <= max_spot and min_vol <= sigma <= max_vol:
                fig_heatmap.add_scatter(
                    x=[S], 
                    y=[sigma * 100], 
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='x'),
                    name='Current Position',
                    hovertemplate=f'Current Position<br>Spot: ${S}<br>Volatility: {sigma*100:.1f}%<extra></extra>'
                )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Heatmap statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Price", f"${np.min(heatmap_data):.4f}")
            with col2:
                st.metric("Max Price", f"${np.max(heatmap_data):.4f}")
            with col3:
                st.metric("Mean Price", f"${np.mean(heatmap_data):.4f}")
            with col4:
                st.metric("Std Dev", f"${np.std(heatmap_data):.4f}")

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
This calculator uses the Black-Scholes model to price European options.

**Custom Price Features:**
- Input actual market prices for comparison
- Get trading recommendations based on mispricing
- Analyze percentage differences between theoretical and market prices

**Heatmap Features:**
- **Green**: Higher option values
- **Red**: Lower option values
- **Blue X**: Current position (if within range)
- Interactive hover for precise values

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
