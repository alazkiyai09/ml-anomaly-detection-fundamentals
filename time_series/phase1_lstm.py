import csv
import os
import requests
import pandas as pd
import psycopg2
from io import StringIO
from psycopg2 import sql
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from telegram import Bot
import asyncio
import numpy as np
import talib
import mplfinance as mpf
import matplotlib.dates as mdates
import arch
from matplotlib.dates import DateFormatter
import yfinance as yf
import json


def test():
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    apikey = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=gc&apikey={apikey}'
    r = requests.get(url)
    data = r.json()

    print(data)

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
def get_daily_data(symbol):

    apikey = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    CSV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={apikey}&datatype=csv'

    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)

    return my_list

def get_fundamental_data(symbol, stats):
    apikey = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    CSV_URL = f'https://www.alphavantage.co/query?function={stats}&symbol={symbol}&apikey={apikey}'
    r = requests.get(CSV_URL)
    data = r.json()
    
    return data

def process_income_data(data_dict):

    # Extract symbol and annual report data
    symbol = data_dict['symbol']
    annual_report_data = data_dict['annualReports']

    # Create a pandas DataFrame
    df = pd.DataFrame(annual_report_data)

    # Convert numeric columns to appropriate data types
    numeric_columns = [
        'grossProfit',
        'totalRevenue',
        'costOfRevenue',
        'costofGoodsAndServicesSold',
        'operatingIncome',
        'sellingGeneralAndAdministrative',
        'researchAndDevelopment',
        'operatingExpenses',
        'investmentIncomeNet',
        'netInterestIncome',
        'interestIncome',
        'interestExpense',
        'otherNonOperatingIncome',
        'depreciation',
        'depreciationAndAmortization',
        'incomeBeforeTax',
        'incomeTaxExpense',
        'interestAndDebtExpense',
        'netIncomeFromContinuingOperations',
        'comprehensiveIncomeNetOfTax',
        'ebit',
        'ebitda',
        'netIncome'
    ]

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric) / 10**9

    # Set fiscalDateEnding as the index
    df.set_index('fiscalDateEnding', inplace=True)

    # Print the DataFrame
    return df

def process_cashflow_data(data_dict):
    # Extract symbol and annual report data
    symbol = data_dict['symbol']
    annual_report_data = data_dict['annualReports']

    # Create a pandas DataFrame
    df = pd.DataFrame(annual_report_data)

    # Convert numeric columns to appropriate data types
    numeric_columns = [
        'operatingCashflow',
        'paymentsForOperatingActivities',
        'changeInOperatingLiabilities',
        'changeInOperatingAssets',
        'depreciationDepletionAndAmortization',
        'capitalExpenditures',
        'changeInReceivables',
        'changeInInventory',
        'profitLoss',
        'cashflowFromInvestment',
        'cashflowFromFinancing',
        'proceedsFromRepaymentsOfShortTermDebt',
        'paymentsForRepurchaseOfCommonStock',
        'paymentsForRepurchaseOfEquity',
        'dividendPayout',
        'dividendPayoutCommonStock',
        'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet',
        'proceedsFromRepurchaseOfEquity',
        'changeInCashAndCashEquivalents',
        'netIncome'
    ]

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric) / 10**9

    # Set fiscalDateEnding as the index
    df.set_index('fiscalDateEnding', inplace=True)

    return df

def process_earning_data(data_dict):
    symbol = data_dict['symbol']
    annual_earning_data = data_dict['annualEarnings']
    df = pd.DataFrame(annual_earning_data)
    # Convert numeric columns to appropriate data types
    numeric_columns = ['reportedEPS']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Set fiscalDateEnding as the index
    df.set_index('fiscalDateEnding', inplace=True)

    return df

def get_data_yfinnace(symbol):
    symbol_data = yf.download(symbol, period="max", interval="1d")
    symbol_data = symbol_data.reset_index()

    column_mapping = {
    'Date': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adjusted_close',
    'Volume': 'volume'
    }

    # Rename the columns using the dictionary
    symbol_data = symbol_data.rename(columns=column_mapping)
    return symbol_data


def get_daily_dataUST(maturity):

    apikey = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    CSV_URL = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity={maturity}&apikey={apikey}&datatype=csv'

    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)

    return my_list


def convert_pandas(data):

    feature_list= data[0]
    price_data = pd.DataFrame(data[1:],columns=feature_list)
    price_data = price_data.iloc[::-1].reset_index(drop=True) 

    return price_data

def create_connection():
    # SECURITY WARNING: Use environment variables for credentials in production
    # Example: password = os.getenv('DB_PASSWORD')
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'Stock_Data'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD')  # Required: must be set in environment
    )
    return conn

def insert_table(data, table_name):
    # Establish a connection to the PostgreSQL database
    conn = create_connection()
    cursor = conn.cursor()

    # Create a string buffer to hold the DataFrame data
    data_buffer = StringIO()
    data.to_csv(data_buffer, index=False, header=False, sep='\t')  # Use appropriate separator if needed

    # Reset the buffer's position to the beginning
    data_buffer.seek(0)

    # Copy the data from the buffer to the PostgreSQL table
    cursor.copy_from(data_buffer, table_name, null='', columns=data.columns)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and the database connection
    cursor.close()
    conn.close()
    return 0

def read_table(table_name):
    # Establish a connection to the PostgreSQL database
    conn = create_connection()
    cursor = conn.cursor()

    # Define the table name in the PostgreSQL database

    # Execute a SELECT query to fetch the data from the table (safe from SQL injection)
    cursor.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name)))

    # Fetch all the rows from the result set
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    columns = [desc[0] for desc in cursor.description]

    # Create a DataFrame from the fetched data and column names
    data = pd.DataFrame(rows, columns=columns)

    # Print the DataFrame or perform further operations

    # Close the cursor and the database connection
    cursor.close()
    conn.close()

    return data

def insert_latest_database(new_record, table_name):
    # Establish a connection to the PostgreSQL database
    conn = create_connection()
    cursor = conn.cursor()

    # Execute a SELECT query to get the last date in the table (safe from SQL injection)
    cursor.execute(sql.SQL("SELECT MAX(Timestamp) FROM {}").format(sql.Identifier(table_name)))

    # Fetch the last date from the result set
    last_date = cursor.fetchone()[0]

    # Convert the 'Timestamp' column of the DataFrame to datetime
    new_record['timestamp'] = pd.to_datetime(new_record['timestamp'])

    # Filter the DataFrame based on the last date
    if last_date is not None:
        data = new_record[(new_record['timestamp'] > last_date)]
        data.loc[:, 'timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert timestamp to string format

        # Check if there are new records to insert
        if not data.empty:
            # Insert the records into the table
            columns = ', '.join(data.columns)  # Get the column names as a comma-separated string
            values = ', '.join([f"('{row['timestamp']}', {', '.join(map(str, row.values[1:]))})" for _, row in data.iterrows()])  # Create values string
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES {values}"
            cursor.execute(insert_query)

            # Commit the changes to the database
            conn.commit()
            print(f"{len(data)} records inserted.")
        else:
            print("New record has an older or same date as the last record in the table. Skipping insertion.")
    else:
        insert_table(new_record, table_name)

    # Close the cursor and the database connection
    cursor.close()
    conn.close()



def delete_xrecords_database(table_name, num_rows_to_delete):
    # Establish a connection to the PostgreSQL database
    conn = create_connection()
    cursor = conn.cursor()


    # Construct the DELETE query to delete the specified number of rows from the table
    delete_query = f"""
        DELETE FROM {table_name}
        WHERE timestamp IN (
            SELECT timestamp
            FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT {num_rows_to_delete}
        )
    """

    # Execute the DELETE query
    cursor.execute(delete_query)

    # Commit the changes to the database
    conn.commit()
    print(f"{num_rows_to_delete} rows deleted successfully.")

    # Close the cursor and the database connection
    cursor.close()
    conn.close()

def create_new_daily_table(new_table_name, id=1):
    conn = create_connection()
    cursor = conn.cursor()

    # Define the table names in the PostgreSQL database
    if id == 1:
        existing_table_name = "stock_daily_data"
    elif id == 2:
        existing_table_name = "daily_treasury_data"
    elif id == 3:
        existing_table_name = "yfinance_data"

    # Check if the table already exists (safe from SQL injection)
    cursor.execute(
        sql.SQL("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = {})")
        .format(sql.Literal(new_table_name))
    )
    table_exists = cursor.fetchone()[0]

    if table_exists:
        print(f"Table '{new_table_name}' already exists. Please choose a different table name.")
        return False
    else:
    
        # Get the column names and data types from the existing table (safe from SQL injection)
        cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(sql.Identifier(existing_table_name)))
        columns = [desc[0] for desc in cursor.description]
        oid_data_types = [desc[1] for desc in cursor.description]

        # Define a mapping from OID data types to their corresponding names
        data_type_mapping = {
            16: "boolean",
            20: "bigint",
            21: "smallint",
            23: "integer",
            25: "text",
            700: "real",
            701: "double precision",
            1114: "timestamp without time zone",
            1700: "numeric",
            # Add more mappings as needed for other data types
        }

        # Map the OID data types to their names and handle unknown types
        data_types = []
        for oid in oid_data_types:
            data_type = data_type_mapping.get(oid)
            if data_type is None:
                raise ValueError(f"Unknown data type for OID: {oid}")
            data_types.append(data_type)

        # Construct the column definitions for the CREATE TABLE query
        column_definitions = [
            sql.SQL('{} {}').format(sql.Identifier(column), sql.SQL(data_type))
            for column, data_type in zip(columns, data_types)
        ]

        # Construct the CREATE TABLE query with the column definitions
        create_table_query = sql.SQL("""
            CREATE TABLE {} (
                {}
            )
        """).format(
            sql.Identifier(new_table_name),
            sql.SQL(',\n').join(column_definitions)
        )

        # Execute the CREATE TABLE query
        cursor.execute(create_table_query)

        # Commit the changes to the database
        conn.commit()
        print(f"Table '{new_table_name}' created successfully.")
        

        # Close the cursor and the database connection
        cursor.close()
        conn.close()
        return True


def check_list_stock():
    conn = create_connection()
    cursor = conn.cursor()
    # Execute the SQL query to fetch all table names
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")

    # Fetch all the table names from the cursor
    table_names = cursor.fetchall()

    # Print the table names
    for name in table_names:
        print(name[0])

    # Close the cursor and the database connection
    cursor.close()
    conn.close()

def add_new_stock_list():

    New_stock = ['XAU', 'DX-Y.NYB', '^VIX','AAPL', 'MSFT', 'CVX', 'TSLA', 'AMD', 'FTNT', 'PANW', 'JNJ', 'PG', 'MA', 'V', 'XOM', 'NVDA']

    for stock in New_stock:
        if stock not in ['^VIX', 'DX-Y.NYB', 'GC=F']:
            new_table_name = stock.lower() +"_daily_data"
            create_new_daily_table(new_table_name)
            temp_data = get_daily_data(stock)
            temp_data = convert_pandas(temp_data)
            print(temp_data.tail(10))
            print(len(temp_data))
            insert_latest_database(temp_data, new_table_name)
            print(read_table(new_table_name))
            time.sleep(30)
        else:
            new_table_name = stock.lower() +"_daily_data"
            new_table_name = new_table_name.replace('.', '')
            new_table_name = new_table_name.replace('-', '')
            new_table_name = new_table_name.replace('=', '')
            create_new_daily_table(new_table_name.replace('^', ''), 3)
            temp_data = get_data_yfinnace(stock)
            print(temp_data.tail(10))
            print(len(temp_data))
            insert_latest_database(temp_data, new_table_name.replace('^', ''))
            print(read_table(new_table_name.replace('^', '')))
            time.sleep(30)

def update_stock_data():

    New_stock = ['GC=F', 'DX-Y.NYB', '^VIX','AAPL', 'MSFT', 'CVX', 'TSLA', 'AMD', 'FTNT', 'PANW', 'JNJ', 'PG', 'MA', 'V', 'XOM', 'NVDA']

    for stock in New_stock:
        if stock not in ['^VIX', 'DX-Y.NYB', 'GC=F']:
            new_table_name = stock.lower() +"_daily_data"
            temp_data = get_daily_data(stock)
            temp_data = convert_pandas(temp_data)
            print(temp_data.head(10))
            print(temp_data.tail(10))
            print(len(temp_data))
            print(stock)
            insert_latest_database(temp_data, new_table_name)
            time.sleep(30)
        else:
            new_table_name = stock.lower() +"_daily_data"
            new_table_name = new_table_name.replace('.', '')
            new_table_name = new_table_name.replace('-', '')
            new_table_name = new_table_name.replace('=', '')
            temp_data = get_data_yfinnace(stock)
            print(temp_data.tail(10))
            print(len(temp_data))
            print(stock)
            insert_latest_database(temp_data, new_table_name.replace('^', ''))
            print(read_table(new_table_name.replace('^', '')))
            time.sleep(30)

def add_new_treasury_list():

    New_treasury = ['3month', '2year', '5year', '7year', '10year', '30year']

    for maturity in New_treasury:
        new_table_name = "treasury" + maturity.lower() +"_daily_data"
        create_new_daily_table(new_table_name, 2)
        temp_data = get_daily_dataUST(maturity)
        temp_data = convert_pandas(temp_data)
        temp_data = temp_data.drop(temp_data[temp_data['value'] == "."].index)
        print(temp_data.tail(10))
        print(len(temp_data))
        insert_latest_database(temp_data, new_table_name)
        print(read_table(new_table_name))
        time.sleep(30)

def update_treasury_data():

    New_treasury = ['3month', '2year', '5year', '7year', '10year', '30year']

    for maturity in New_treasury:
        new_table_name = "treasury" + maturity.lower() +"_daily_data"
        temp_data = get_daily_dataUST(maturity)
        temp_data = convert_pandas(temp_data)
        temp_data = temp_data.drop(temp_data[temp_data['value'] == "."].index)
        print(temp_data.tail(10))
        print(len(temp_data))
        insert_latest_database(temp_data, new_table_name)
        time.sleep(30)

def print_graph(stock_name, n=None):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)
    # Convert 'timestamp' column to datetime format if needed
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
    if n is None:
        stock_data = stock_data
    else:
        stock_data = stock_data.tail(n)

    # Create figure
    fig = go.Figure()

    # Add stock price trace
    fig.add_trace(go.Scatter(
        x=stock_data['timestamp'],
        y=stock_data['adjusted_close'],
        mode='lines',
        name='Price',
        hovertemplate='Date: %{x}<br>Price: $%{y:.3f}<extra></extra>'
    ))

    # Add volume trace as a bar chart
    fig.add_trace(go.Bar(
        x=stock_data['timestamp'],
        y=stock_data['volume'],
        name='Volume',
        hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>',
        yaxis="y2"
    ))

    # Set layout
    fig.update_layout(
        title=stock_name + ' Stock Price and Trading Volume Over Time',
        xaxis_title='Date',
        yaxis=dict(title='Price'),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        hovermode='x'
    )

    # Save the plot as a PNG image
    image_filename = stock_name + '_stock_price_volume.png'
    pio.write_image(fig, image_filename, format='png')
    # Display the plot
    fig.show()

async def send_telegram(filename):
    # SECURITY WARNING: Use environment variables for credentials in production
    # Get Telegram credentials from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    channel_id = os.getenv('TELEGRAM_CHANNEL_ID')

    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")
    if not channel_id:
        raise ValueError("TELEGRAM_CHANNEL_ID environment variable not set")

    # Message text
    message_text = 'Check out '+filename.replace('.png','')

    # Send the image and message to the Telegram channel
    bot = Bot(token=bot_token)
    with open(filename, 'rb') as image:
        await bot.send_photo(chat_id=channel_id, photo=image, caption=message_text)

def interactive_graph(stock_name, n=None):
    table_name = stock_name.lower() +"_daily_data"
    stock_data = read_table(table_name)
    # Convert 'Date' column to datetime format if needed
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
    if n is None:
        stock_data = stock_data
    else:
        stock_data = stock_data.tail(n)
    # Create subplots with shared x-axis and different y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the stock price trace
    fig.add_trace(go.Scatter(x=stock_data['timestamp'], y=stock_data['adjusted_close'], mode='lines', name='Stock Price',
                            hovertemplate='Date: %{x}<br>Price: $%{customdata[0]:.3f}<br>Volume: %{customdata[1]:,}<extra></extra>',
                            hovertext=stock_data[['adjusted_close', 'volume']],
                            customdata=stock_data[['adjusted_close', 'volume']]))

    # Add the trading volume trace
    fig.add_trace(go.Bar(x=stock_data['timestamp'], y=stock_data['volume'], name='Volume',
                        hovertemplate='Date: %{x}<br>Volume: %{y:,}<extra></extra>'), secondary_y=True)

    # Customize the layout
    fig.update_layout(title=stock_name+' Stock Price and Trading Volume Over Time', xaxis_title='Date')

    # Set y-axis titles
    fig.update_yaxes(title_text='Price and Volume', secondary_y=False)

    # Display the interactive plot
    fig.show()


def ta_stock(stock_name):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)

    stock_data = stock_data.tail(90)

    stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')

    # Convert the index to a DatetimeIndex
    stock_data.index = pd.to_datetime(stock_data.index)

    # Calculate Moving Average Convergence Divergence (MACD)
    macd, macd_signal, macd_hist = talib.MACD(stock_data['adjusted_close'])

    # Calculate Bollinger Bands
    upper_band, middle_band, lower_band = talib.BBANDS(stock_data['adjusted_close'])

    # Calculate Simple Moving Averages (SMA)
    sma_short = talib.SMA(stock_data['adjusted_close'], timeperiod=10)
    sma_long = talib.SMA(stock_data['adjusted_close'], timeperiod=20)

    # Combine indicators into a DataFrame
    indicators = pd.DataFrame({
        'MACD': macd,
        'MACD Signal': macd_signal,
        'MACD Histogram': macd_hist,
        'Upper Bollinger Band': upper_band,
        'Middle Bollinger Band': middle_band,
        'Lower Bollinger Band': lower_band,
        'Short-term SMA': sma_short,
        'Long-term SMA': sma_long
    }, index=stock_data.index)

    # Plot candlestick chart with indicators
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot candlestick chart
    mpf.plot(stock_data, ax=ax1, type='candle', style='yahoo', title='Stock Chart')

    # Plot indicators
    ax2.plot(indicators.index, indicators['MACD'], label='MACD')
    ax2.plot(indicators.index, indicators['MACD Signal'], label='MACD Signal')
    ax2.bar(indicators.index, indicators['MACD Histogram'], label='MACD Histogram')
    ax2.plot(indicators.index, indicators['Upper Bollinger Band'], label='Upper Bollinger Band')
    ax2.plot(indicators.index, indicators['Middle Bollinger Band'], label='Middle Bollinger Band')
    ax2.plot(indicators.index, indicators['Lower Bollinger Band'], label='Lower Bollinger Band')
    ax2.plot(indicators.index, indicators['Short-term SMA'], label='Short-term SMA')
    ax2.plot(indicators.index, indicators['Long-term SMA'], label='Long-term SMA')

    ax2.set_ylabel('Indicator Values')
    ax2.legend()

    plt.show()

def sma_indicator(stock_name, n=None):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)

    if n is None:
        n = stock_data.shape[0]
    else:
        n = n
    # Convert the necessary columns to numeric
    stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])

    stock_data.set_index('timestamp', inplace=True)

    # Calculate SMAs
    stock_data['SMA'] = stock_data['adjusted_close'].rolling(window=20).mean()
    stock_data['SMA50'] = stock_data['adjusted_close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['adjusted_close'].rolling(window=200).mean()

    # Plotting
    stock_data = stock_data.tail(n)
    fig, axes = mpf.plot(stock_data, type='candle', style='yahoo', title=stock_name+' Stock Chart with SMA indicator',
                        mav=(20, 50, 200), returnfig=True)

    legend_labels = ['Candlestick', 'SMA', 'SMA50', 'SMA200']
    axes[0].legend(legend_labels)

    save_file_name = stock_name+'Stock_chart_with_SMA.png'
    plt.savefig(save_file_name)

    plt.show()


def ema_indicator(stock_name, n=None):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)
    if n is None:
        n = stock_data.shape[0]
    else:
        n = n

    # Convert the necessary columns to numeric
    stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])

    stock_data.set_index('timestamp', inplace=True)

    # Calculate EMAs
    stock_data['EMA20'] = talib.EMA(stock_data['adjusted_close'], timeperiod = 20)
    stock_data['EMA50'] = talib.EMA(stock_data['adjusted_close'], timeperiod = 50)
    stock_data['EMA200'] = talib.EMA(stock_data['adjusted_close'], timeperiod = 200)


    # Plotting
    stock_data = stock_data.tail(n)
    fig, axes = mpf.plot(stock_data, type='candle', style='yahoo', title=stock_name+' Stock Chart with EMA indicator',
                        mav=(10,50, 200,), returnfig=True)

    legend_labels = ['Candlestick', 'EMA', 'EMA50', 'EMA200']
    axes[0].legend(legend_labels)

    save_file_name = stock_name+'Stock_chart_with_EMA.png'
    plt.savefig(save_file_name)

    plt.show()

def adx_indicator(stock_name, n=None):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)

    if n is None:
        n = stock_data.shape[0]
    else:
        n = n

    # Convert the necessary columns to numeric
    stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])

    stock_data.set_index('timestamp', inplace=True)

    # Calculate ADX
    stock_data['ADX'] = talib.ADX(stock_data['high'], stock_data['low'], stock_data['close'], timeperiod=20)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    stock_data = stock_data.tail(n)
    # Plot the candlestick chart on subplot 1
    mpf.plot(stock_data, type='candle', style='yahoo', ax=ax1)
    ax1.set_title('Stock Chart')
    ax1.set_ylabel('Price')

    # Plot the ADX chart on subplot 2
    ax2.plot(stock_data.index, stock_data['ADX'].tail(n), label='ADX', color='orange')
    ax2.axhline(25, color='red', linestyle='--')  # Add ADX threshold at 25
    ax2.set_title('ADX Chart')
    ax2.set_ylabel('ADX')
    ax2.set_xlabel('Date')

    # Set x-axis limits for the ADX subplot
    ax2.set_xlim(stock_data.index[0], stock_data.index[-1])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the charts
    plt.show()


def rsi_indicator(stock_name, n=None):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)

    if n is None:
        n = stock_data.shape[0]
    else:
        n = n

    # Convert the necessary columns to numeric
    stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])

    stock_data.set_index('timestamp', inplace=True)

    # Calculate RSI
    stock_data['RSI'] = talib.RSI(stock_data['adjusted_close'], timeperiod=14)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    stock_data = stock_data.tail(n)
    # Plot the candlestick chart on subplot 1
    mpf.plot(stock_data, type='candle', style='yahoo', ax=ax1)
    ax1.set_title('Stock Chart')
    ax1.set_ylabel('Price')

    # Plot the RSI chart on subplot 2
    ax2.plot(stock_data.index, stock_data['RSI'].tail(n), label='RSI', color='orange')
    ax2.axhline(30, color='red', linestyle='--')  # Add RSI oversold level at 30
    ax2.axhline(70, color='green', linestyle='--')  # Add RSI overbought level at 70
    ax2.set_ylabel('RSI')

    # Set x-axis limits for the RSI subplot
    ax2.set_xlim(stock_data.index[0], stock_data.index[-1])

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the chart
    plt.show()


def bollinger_indicator(stock_name, n=None):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)

    if n is None:
        n = stock_data.shape[0]
    else:
        n = n

    # Convert the necessary columns to numeric
    stock_data['open'] = pd.to_numeric(stock_data['open'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])

    stock_data.set_index('timestamp', inplace=True)

    # Calculate Bollinger Bands
    stock_data['upper'], stock_data['middle'], stock_data['lower'] = talib.BBANDS(stock_data['adjusted_close'], timeperiod=20)


    # Plotting
    stock_data = stock_data.tail(n)
    fig, axes = mpf.plot(stock_data, type='candle', style='yahoo', addplot=mpf.make_addplot(stock_data[['upper', 'middle', 'lower']]),
                     returnfig=True)
    
    save_file_name = stock_name+'Stock_chart_with_Bollinger.png'
    plt.savefig(save_file_name)

    plt.show()


def show_volatility(stock_name, n=None):
    # Read stock price data
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)
    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])

    # Read VIX data
    vix_data = read_table("vix_daily_data")
    vix_data['close'] = pd.to_numeric(vix_data['close'], errors='coerce')
    vix_data['timestamp'] = pd.to_datetime(vix_data['timestamp'])

    # Read economic data (e.g., bond yields)
    bond_data = read_table("treasury2year_daily_data")
    bond_data['value'] = pd.to_numeric(bond_data['value'], errors='coerce')
    bond_data['timestamp'] = pd.to_datetime(bond_data['timestamp'])

    # Set index to timestamp
    stock_data.set_index('timestamp', inplace=True)
    vix_data.set_index('timestamp', inplace=True)
    bond_data.set_index('timestamp', inplace=True)

    # Calculate returns based on stock prices
    stock_returns = np.log(stock_data['adjusted_close'] / stock_data['adjusted_close'].shift(1)).dropna()
    bond_returns = np.log(bond_data['value']).diff().dropna()

    # Combine the returns into a single Series
    returns = pd.concat([stock_returns, bond_returns], axis=1).dropna().iloc[:, 0]

    garch_models = [
        ('GARCH', arch.arch_model(returns, vol='Garch', p=1, q=1, rescale=False)),
        ('EGARCH', arch.arch_model(returns, vol='EGarch', p=1, q=1, rescale=False)),
        ('GJR-GARCH', arch.arch_model(returns, vol='Garch', p=1, o=1, q=1, rescale=False)),
        ('Multivariate GARCH', arch.arch_model(returns, vol='Garch', p=1, q=1, rescale=False, dist='normal'))
    ]

    garch_forecasts = []

    for model_name, model in garch_models:
        model_fit = model.fit()
        forecast_volatility = model_fit.forecast(start=0, reindex=True)
        forecast_volatility =  np.sqrt(forecast_volatility.variance.values)
        garch_forecasts.append((model_name, forecast_volatility))

    if n is None:
        n = min(stock_data.shape[0],vix_data.shape[0], bond_data.shape[0], len(forecast_volatility))
    else:
        n = n
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))

    # Plot the stock prices in the first subplot
    ax1.plot(stock_data.index[-n:], stock_data['adjusted_close'].tail(n), color='blue', label='Stock Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Stock Price')
    ax1.legend(loc='upper left')

    # Format the x-axis as dates
    date_formatter = DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()

    # Plot the VIX data in the second subplot
    ax2.plot(vix_data.index[-n:], vix_data['close'].tail(n), color='purple', label='VIX')
    ax2.set_ylabel('VIX')
    ax2.set_title('VIX')
    ax2.legend(loc='upper left')

    # Format the x-axis as dates
    ax2.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()

    # Plot the forecasted volatilities for each GARCH model and Multivariate GARCH in the third subplot
    colors = ['red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', ':', '-.']
    opacities = [1.0, 0.8, 0.6, 1.0]

    for i, (name, forecast_volatility) in enumerate(garch_forecasts):
        forecast_volatility = forecast_volatility[-n:]
        ax3.plot(stock_data.index[-n:], np.sqrt(forecast_volatility),
                color=colors[i], linestyle=linestyles[i], alpha=opacities[i], label=name)

    ax3.set_ylabel('Volatility')
    ax3.set_title('GARCH Models Comparison')
    ax3.legend(loc='upper left')

    # Format the x-axis as dates
    ax3.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()
                    

async def main():
    #check_list_stock()
    #print_graph('NVDA', 30)
    #interactive_graph('NVDA')
    stock_name = 'NVDA'
    #ta_stock('NVDA')
    #sma_indicator(stock_name, 360)
    #ema_indicator(stock_name, 360)
    #adx_indicator(stock_name, 360)
    #rsi_indicator(stock_name, 60)
    #await send_telegram(stock_name+'Stock_chart_with_SMA.png')
    #bollinger_indicator(stock_name, 360)
    #show_volatility(stock_name, 360)
    #update_stock_data()
    #update_treasury_data()
    #data = get_fundamental_data(stock_name, 'INCOME_STATEMENT')
    #data = process_income_data(data)
    #add_new_stock_list()
    test()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())