import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import os
import numpy as np

# Function to fetch stock indicators
def fetch_indicators(stock, interval='1d'):
    ticker = yf.Ticker(stock)
    data = ticker.history(period="1y", interval=interval)

    if data.empty or len(data) < 2:
        return {key: None for key in [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Upper_BB', 'Lower_BB', 'Volatility', 'Beta',
            'Close', 'Volume', 'SMA_50', 'SMA_200', 
            'EMA_12', 'EMA_26', 'Average_Volume', 
            'Average_Volume_10d', 'Pattern', 
            'Strength_Percentage', 'Bullish_Percentage', 'Bearish_Percentage',
            'Support_Level', 'Resistance_Level', 'PE_Ratio', 'PB_Ratio',
            'Dividend_Payout_Ratio', 'EPS', 'Debt_to_Equity', 'Promoter_Holding'
        ]}

    # Calculate technical indicators
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Upper_BB'] = bb.bollinger_hband()
    data['Lower_BB'] = bb.bollinger_lband()
    data['Volatility'] = data['Close'].pct_change().rolling(window=21).std() * 100
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()

    # Calculate Support and Resistance Levels
    data['Support_Level'] = data['Close'].rolling(window=20).min()
    data['Resistance_Level'] = data['Close'].rolling(window=20).max()

    # Get fundamental data
    info = ticker.info
    pe_ratio = info.get('trailingPE', None)
    pb_ratio = info.get('priceToBook', None)
    dividend_payout = info.get('payoutRatio', None)
    eps = info.get('trailingEps', None)
    debt_to_equity = info.get('debtToEquity', None)
    promoter_holding = info.get('heldPercentInsiders', None)
    
    # If promoter holding not found, try other keys
    if promoter_holding is None:
        promoter_holding = info.get('majorityHoldings', None)
    
    # Convert to percentage if promoter holding is found
    if promoter_holding is not None and promoter_holding < 1:
        promoter_holding = promoter_holding * 100

    average_volume = data['Volume'].mean()
    average_volume_10d = data['Volume'].rolling(window=10).mean().iloc[-1] if len(data['Volume']) >= 10 else None
    beta = ticker.info.get('beta', None)

    last_close = data['Close'].iloc[-1]
    pattern = detect_chart_pattern(data)

    return {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'MACD_Signal': data['MACD_Signal'].iloc[-1],
        'MACD_Hist': data['MACD_Hist'].iloc[-1],
        'Upper_BB': data['Upper_BB'].iloc[-1],
        'Lower_BB': data['Lower_BB'].iloc[-1],
        'Volatility': data['Volatility'].iloc[-1],
        'Beta': beta,
        'Close': last_close,
        'Volume': data['Volume'].iloc[-1],
        'SMA_50': data['SMA_50'].iloc[-1],
        'SMA_200': data['SMA_200'].iloc[-1],
        'EMA_12': data['EMA_12'].iloc[-1],
        'EMA_26': data['EMA_26'].iloc[-1],
        'Average_Volume': average_volume,
        'Average_Volume_10d': average_volume_10d,
        'Pattern': pattern,
        'Strength_Percentage': ((last_close - data['SMA_50'].iloc[-1]) / data['SMA_50'].iloc[-1] * 100) if data['SMA_50'].iloc[-1] is not None else 0,
        'Bullish_Percentage': calculate_bullish_percentage(data),
        'Bearish_Percentage': calculate_bearish_percentage(data),
        'Support_Level': data['Support_Level'].iloc[-1],
        'Resistance_Level': data['Resistance_Level'].iloc[-1],
        'PE_Ratio': pe_ratio,
        'PB_Ratio': pb_ratio,
        'Dividend_Payout_Ratio': dividend_payout,
        'EPS': eps,
        'Debt_to_Equity': debt_to_equity,
        'Promoter_Holding': promoter_holding
    }

# Function to detect chart patterns (unchanged from previous version)
def detect_chart_pattern(data):
    if len(data) < 30:
        return "No Pattern"
    # ... (rest of the detect_chart_pattern function remains the same)
    return recognized_patterns if recognized_patterns else ["No Recognized Pattern"]

# Pattern detection helper functions (unchanged from previous version)
def is_head_and_shoulders(prices):
    # ... (implementation remains the same)
    return len(peak_indices) >= 2 and len(valley_indices) >= 1

def is_double_top(prices):
    # ... (implementation remains the same)
    return len(peak_indices) >= 2 and abs(prices[peak_indices[0]] - prices[peak_indices[1]]) < 0.01 * prices[peak_indices[0]]

def is_double_bottom(prices):
    # ... (implementation remains the same)
    return len(valley_indices) >= 2 and abs(prices[valley_indices[0]] - prices[valley_indices[1]]) < 0.01 * prices[valley_indices[0]]

def is_symmetrical_triangle(prices):
    # ... (implementation remains the same)
    return (prices[peak_indices[-1]] < prices[peak_indices[0]]) and (prices[valley_indices[-1]] > prices[valley_indices[0]])

def is_ascending_triangle(prices):
    # ... (implementation remains the same)
    return (len(peak_indices) >= 2 and len(valley_indices) >= 2 and
            prices[valley_indices[-1]] > prices[valley_indices[0]] and
            prices[peak_indices[-1]] < prices[peak_indices[0]])

def is_descending_triangle(prices):
    # ... (implementation remains the same)
    return (len(peak_indices) >= 2 and len(valley_indices) >= 2 and
            prices[valley_indices[-1]] < prices[valley_indices[0]] and
            prices[peak_indices[-1]] > prices[peak_indices[0]])

# Bullish/Bearish percentage calculations (unchanged from previous version)
def calculate_bullish_percentage(data):
    bullish_count = sum(data['Close'].diff().dropna() > 0)
    total_count = len(data) - 1
    return (bullish_count / total_count * 100) if total_count > 0 else 0

def calculate_bearish_percentage(data):
    bearish_count = sum(data['Close'].diff().dropna() < 0)
    total_count = len(data) - 1
    return (bearish_count / total_count * 100) if total_count > 0 else 0

# Function to score stocks based on indicators for different terms
def score_stock(indicators, term):
    score = 0

    if term == 'Short Term':
        if indicators['RSI'] is not None:
            if indicators['RSI'] < 30 or indicators['RSI'] > 70:
                score += 2
            if 30 <= indicators['RSI'] <= 40 or 60 <= indicators['RSI'] <= 70:
                score += 1

        if indicators['MACD'] is not None:
            if indicators['MACD'] > 0 and indicators['MACD'] > indicators['MACD_Signal']:
                score += 2

        # Additional scoring for short term
        if indicators['Close'] > indicators['Support_Level']:
            score += 1
        if indicators['Close'] < indicators['Resistance_Level']:
            score += 1

    elif term == 'Medium Term':
        if indicators['RSI'] is not None:
            if 40 <= indicators['RSI'] <= 60:
                score += 2

        if indicators['MACD'] is not None:
            if abs(indicators['MACD']) < 0.01:
                score += 1

        # Additional scoring for medium term
        if indicators['PE_Ratio'] is not None and indicators['PE_Ratio'] < 25:
            score += 1
        if indicators['Debt_to_Equity'] is not None and indicators['Debt_to_Equity'] < 1:
            score += 1

    elif term == 'Long Term':
        if indicators['RSI'] is not None:
            if 40 <= indicators['RSI'] <= 60:
                score += 2

        if indicators['Beta'] is not None:
            if 0.9 <= indicators['Beta'] <= 1.1:
                score += 2

        # Additional scoring for long term
        if indicators['PE_Ratio'] is not None and indicators['PE_Ratio'] < 20:
            score += 1
        if indicators['PB_Ratio'] is not None and indicators['PB_Ratio'] < 3:
            score += 1
        if indicators['Promoter_Holding'] is not None and indicators['Promoter_Holding'] > 40:
            score += 1
        if indicators['Dividend_Payout_Ratio'] is not None and indicators['Dividend_Payout_Ratio'] > 0.3:
            score += 1

    return score

# Function to generate recommendations based on different strategies
def generate_recommendations(indicators_list):
    recommendations = {
        'Short Term': [],
        'Medium Term': [],
        'Long Term': []
    }
    
    for stock, indicators in indicators_list.items():
        current_price = indicators['Close']
        
        if current_price is not None:
            # Entry and exit levels based on support/resistance
            entry_level = indicators['Support_Level'] if indicators['Support_Level'] is not None else current_price * 0.99
            exit_level = indicators['Resistance_Level'] if indicators['Resistance_Level'] is not None else current_price * 1.05
            
            # Stop loss and target calculations
            short_stop_loss = current_price * (1 - 0.03)
            short_target = current_price * (1 + 0.05)
            medium_stop_loss = current_price * (1 - 0.04)
            medium_target = current_price * (1 + 0.10)
            long_stop_loss = current_price * (1 - 0.05)
            long_target = current_price * (1 + 0.15)

            short_score = score_stock(indicators, 'Short Term')
            medium_score = score_stock(indicators, 'Medium Term')
            long_score = score_stock(indicators, 'Long Term')

            if short_score > 0:
                recommendations['Short Term'].append({
                    'Stock': stock.replace('.NS', ''),
                    'Current Price': current_price,
                    'Entry Level': entry_level,
                    'Exit Level': exit_level,
                    'Stop Loss': short_stop_loss,
                    'Target Price': short_target,
                    'Score': short_score,
                    'RSI': indicators['RSI'],
                    'MACD': indicators['MACD'],
                    'MACD_Signal': indicators['MACD_Signal'],
                    'Upper_BB': indicators['Upper_BB'],
                    'Lower_BB': indicators['Lower_BB'],
                    'Volatility': indicators['Volatility'],
                    'Beta': indicators['Beta'],
                    'Volume': indicators['Volume'],
                    'SMA_50': indicators['SMA_50'],
                    'SMA_200': indicators['SMA_200'],
                    'EMA_12': indicators['EMA_12'],
                    'EMA_26': indicators['EMA_26'],
                    'Average_Volume': indicators['Average_Volume'],
                    'Average_Volume_10d': indicators['Average_Volume_10d'],
                    'Pattern': indicators['Pattern'],
                    'Strength_Percentage': indicators['Strength_Percentage'],
                    'Bullish_Percentage': indicators['Bullish_Percentage'],
                    'Bearish_Percentage': indicators['Bearish_Percentage'],
                    'Support_Level': indicators['Support_Level'],
                    'Resistance_Level': indicators['Resistance_Level'],
                    'PE_Ratio': indicators['PE_Ratio'],
                    'PB_Ratio': indicators['PB_Ratio'],
                    'Dividend_Payout_Ratio': indicators['Dividend_Payout_Ratio'],
                    'EPS': indicators['EPS'],
                    'Debt_to_Equity': indicators['Debt_to_Equity'],
                    'Promoter_Holding': indicators['Promoter_Holding']
                })

            if medium_score > 0:
                recommendations['Medium Term'].append({
                    'Stock': stock.replace('.NS', ''),
                    'Current Price': current_price,
                    'Entry Level': entry_level,
                    'Exit Level': exit_level,
                    'Stop Loss': medium_stop_loss,
                    'Target Price': medium_target,
                    'Score': medium_score,
                    'RSI': indicators['RSI'],
                    'MACD': indicators['MACD'],
                    'MACD_Signal': indicators['MACD_Signal'],
                    'Upper_BB': indicators['Upper_BB'],
                    'Lower_BB': indicators['Lower_BB'],
                    'Volatility': indicators['Volatility'],
                    'Beta': indicators['Beta'],
                    'Volume': indicators['Volume'],
                    'SMA_50': indicators['SMA_50'],
                    'SMA_200': indicators['SMA_200'],
                    'EMA_12': indicators['EMA_12'],
                    'EMA_26': indicators['EMA_26'],
                    'Average_Volume': indicators['Average_Volume'],
                    'Average_Volume_10d': indicators['Average_Volume_10d'],
                    'Pattern': indicators['Pattern'],
                    'Strength_Percentage': indicators['Strength_Percentage'],
                    'Bullish_Percentage': indicators['Bullish_Percentage'],
                    'Bearish_Percentage': indicators['Bearish_Percentage'],
                    'Support_Level': indicators['Support_Level'],
                    'Resistance_Level': indicators['Resistance_Level'],
                    'PE_Ratio': indicators['PE_Ratio'],
                    'PB_Ratio': indicators['PB_Ratio'],
                    'Dividend_Payout_Ratio': indicators['Dividend_Payout_Ratio'],
                    'EPS': indicators['EPS'],
                    'Debt_to_Equity': indicators['Debt_to_Equity'],
                    'Promoter_Holding': indicators['Promoter_Holding']
                })

            if long_score > 0:
                recommendations['Long Term'].append({
                    'Stock': stock.replace('.NS', ''),
                    'Current Price': current_price,
                    'Entry Level': entry_level,
                    'Exit Level': exit_level,
                    'Stop Loss': long_stop_loss,
                    'Target Price': long_target,
                    'Score': long_score,
                    'RSI': indicators['RSI'],
                    'MACD': indicators['MACD'],
                    'MACD_Signal': indicators['MACD_Signal'],
                    'Upper_BB': indicators['Upper_BB'],
                    'Lower_BB': indicators['Lower_BB'],
                    'Volatility': indicators['Volatility'],
                    'Beta': indicators['Beta'],
                    'Volume': indicators['Volume'],
                    'SMA_50': indicators['SMA_50'],
                    'SMA_200': indicators['SMA_200'],
                    'EMA_12': indicators['EMA_12'],
                    'EMA_26': indicators['EMA_26'],
                    'Average_Volume': indicators['Average_Volume'],
                    'Average_Volume_10d': indicators['Average_Volume_10d'],
                    'Pattern': indicators['Pattern'],
                    'Strength_Percentage': indicators['Strength_Percentage'],
                    'Bullish_Percentage': indicators['Bullish_Percentage'],
                    'Bearish_Percentage': indicators['Bearish_Percentage'],
                    'Support_Level': indicators['Support_Level'],
                    'Resistance_Level': indicators['Resistance_Level'],
                    'PE_Ratio': indicators['PE_Ratio'],
                    'PB_Ratio': indicators['PB_Ratio'],
                    'Dividend_Payout_Ratio': indicators['Dividend_Payout_Ratio'],
                    'EPS': indicators['EPS'],
                    'Debt_to_Equity': indicators['Debt_to_Equity'],
                    'Promoter_Holding': indicators['Promoter_Holding']
                })

    return recommendations

# Main Streamlit application
st.title('Stock Indicator Analysis')

# Check if stocklist.xlsx exists
if os.path.exists('stocklist.xlsx'):
    # Read all sheet names from the Excel file
    xls = pd.ExcelFile('stocklist.xlsx')
    sheet_names = xls.sheet_names
    
    # Let user select which sheet to analyze
    selected_sheet = st.selectbox('Select stock list to analyze:', sheet_names)
    
    # Read the selected sheet
    stock_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
    
    if stock_df.empty or 'Symbol' not in stock_df.columns:
        st.error("The selected sheet does not contain a valid 'Symbol' column.")
        st.stop()

    # Add .NS suffix for NSE stocks if not already present
    stock_symbols = [symbol if '.NS' in symbol else f"{symbol}.NS" for symbol in stock_df['Symbol'].tolist()]
    
    # Display the stocks being analyzed
    st.write(f"Analyzing {len(stock_symbols)} stocks from {selected_sheet}")
    
    # Fetch indicators for all stocks
    indicators_list = {}
    for stock in stock_symbols:
        try:
            indicators = fetch_indicators(stock)
            indicators_list[stock] = indicators
        except Exception as e:
            st.warning(f"Could not fetch data for {stock}: {str(e)}")
            continue

    # Generate recommendations based on the fetched indicators
    recommendations = generate_recommendations(indicators_list)

    if not isinstance(recommendations, dict) or not all(isinstance(val, list) for val in recommendations.values()):
        st.error("No valid recommendations generated.")
        st.stop()

    # Display recommendations as tables
    for term, stocks in recommendations.items():
        st.subheader(f"{term} Recommendations")
        if stocks:
            df = pd.DataFrame(stocks)

            # Round numerical columns to 2 decimals
            numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
            df[numeric_cols] = df[numeric_cols].round(2)

            # Format percentages
            percent_cols = ['RSI', 'Volatility', 'Strength_Percentage', 'Bullish_Percentage', 
                          'Bearish_Percentage', 'Dividend_Payout_Ratio', 'Promoter_Holding']
            for col in percent_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)

            # Check for columns with mixed types or None values and handle them
            for col in df.columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna('N/A')  # Fill NaN with 'N/A'

            st.dataframe(df)

            # Provide download option for the Excel file
            excel_file = f"{selected_sheet}_{term}_recommendations.xlsx"
            with pd.ExcelWriter(excel_file) as writer:
                df.to_excel(writer, index=False, sheet_name=term)

            # Streamlit download button
            with open(excel_file, 'rb') as f:
                st.download_button(
                    label=f"Download {term} Excel file",
                    data=f,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.write("No recommendations available.")
else:
    # Fallback to file upload if stocklist.xlsx doesn't exist
    st.warning("Default stocklist.xlsx file not found. Please upload your own file.")
    
    # Upload file
    uploaded_file = st.file_uploader("Upload a CSV or Excel file with stock symbols", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            stock_df = pd.read_csv(uploaded_file)
        else:
            stock_df = pd.read_excel(uploaded_file)

        if stock_df.empty or 'Stock' not in stock_df.columns:
            st.error("The uploaded file does not contain a valid 'Stock' column.")
            st.stop()

        stock_symbols = stock_df['Stock'].tolist()
        
        # Fetch indicators for all stocks
        indicators_list = {}
        for stock in stock_symbols:
            try:
                indicators = fetch_indicators(stock)
                indicators_list[stock] = indicators
            except Exception as e:
                st.warning(f"Could not fetch data for {stock}: {str(e)}")
                continue

        # Generate recommendations based on the fetched indicators
        recommendations = generate_recommendations(indicators_list)

        if not isinstance(recommendations, dict) or not all(isinstance(val, list) for val in recommendations.values()):
            st.error("No valid recommendations generated.")
            st.stop()

        # Display recommendations as tables
        for term, stocks in recommendations.items():
            st.subheader(f"{term} Recommendations")
            if stocks:
                df = pd.DataFrame(stocks)

                # Round numerical columns to 2 decimals
                numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
                df[numeric_cols] = df[numeric_cols].round(2)

                # Format percentages
                percent_cols = ['RSI', 'Volatility', 'Strength_Percentage', 'Bullish_Percentage', 
                              'Bearish_Percentage', 'Dividend_Payout_Ratio', 'Promoter_Holding']
                for col in percent_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else x)

                # Check for columns with mixed types or None values and handle them
                for col in df.columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna('N/A')

                st.dataframe(df)

                # Provide download option for the Excel file
                excel_file = f"{term}_recommendations.xlsx"
                with pd.ExcelWriter(excel_file) as writer:
                    df.to_excel(writer, index=False, sheet_name=term)

                # Streamlit download button
                with open(excel_file, 'rb') as f:
                    st.download_button(
                        label="Download Excel file",
                        data=f,
                        file_name=excel_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.write("No recommendations available.")
