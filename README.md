# Black-Scholes Option Pricing Calculator with MySQL Integration

This project is a **Streamlit web application** for pricing European options using the Black-Scholes model. It also calculates option Greeks, generates heatmaps, and stores calculation history in a MySQL database.

## Features

- Calculate European call and put option prices using the Black-Scholes formula
- Compute option Greeks (Delta, Gamma, Theta, Vega, Rho)
- Visualize price and volatility sensitivities
- Generate interactive heatmaps for option prices
- Compare theoretical and market prices
- Store and view calculation history in a MySQL database

---

## Requirements

- Python 3.8+
- MySQL Server (8.0+ recommended)
- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Plotly](https://plotly.com/python/)
- [Pandas](https://pandas.pydata.org/)
- [mysql-connector-python](https://pypi.org/project/mysql-connector-python/)

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/option_pricing.git
   cd option_pricing
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   # Or, if requirements.txt is missing:
   pip install streamlit numpy scipy plotly pandas mysql-connector-python
   ```

3. **Install and start MySQL:**
   - On macOS (Homebrew):
     ```sh
     brew install mysql
     brew services start mysql
     ```
   - On Ubuntu:
     ```sh
     sudo apt update
     sudo apt install mysql-server
     sudo systemctl start mysql
     ```

---

## Database Setup

1. **Create the database and tables:**

   Log into MySQL:
   ```sh
   mysql -u root -p
   ```
   Then run:
   ```sql
   CREATE DATABASE IF NOT EXISTS option_pricing;
   USE option_pricing;

   CREATE TABLE IF NOT EXISTS BlackScholesInputs (
       CalculationId INT AUTO_INCREMENT PRIMARY KEY,
       CurrentAssetPrice DOUBLE,
       StrikePrice DOUBLE,
       TimeToMaturity DOUBLE,
       RiskFreeRate DOUBLE,
       Volatility DOUBLE,
       CalculationTimestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE IF NOT EXISTS BlackScholesOutputs (
       OutputId INT AUTO_INCREMENT PRIMARY KEY,
       VolatilityShock DOUBLE,
       StockPriceShock DOUBLE,
       OptionPrice DOUBLE,
       IsCall BOOLEAN,
       CalculationId INT,
       FOREIGN KEY (CalculationId) REFERENCES BlackScholesInputs(CalculationId)
   );
   ```

2. **Configure Streamlit secrets:**

   Create a file at `.streamlit/secrets.toml` in your project directory:
   ```toml
   [mysql]
   host = "localhost"
   user = "root"
   password = "YOUR_MYSQL_ROOT_PASSWORD"
   database = "option_pricing"
   ```
   Replace `YOUR_MYSQL_ROOT_PASSWORD` with your actual MySQL root password.

---

## Running the App

1. **Start the Streamlit app:**
   ```sh
   streamlit run main.py
   ```
2. **Open your browser:**
   Go to [http://localhost:8501](http://localhost:8501)

---

## Usage

- Enter option parameters in the sidebar and click **Calculate Options**
- View theoretical prices, Greeks, and sensitivity charts
- Compare with custom market prices
- Generate and explore option price heatmaps
- View calculation history (if database is connected)

---

## Troubleshooting

- **MySQL connection errors:**
  - Ensure MySQL server is running
  - Check your credentials in `.streamlit/secrets.toml`
  - Make sure the database and tables exist
- **Missing dependencies:**
  - Run `pip install -r requirements.txt` again
- **Port conflicts:**
  - If Streamlit or MySQL won't start, check for other processes using the same port

---

## License

MIT License

---

## Author

- [Dev Rathore](https://github.com/superdev12) 