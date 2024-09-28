from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import requests 
import pandas as pd
import csv 
from pycoingecko import CoinGeckoAPI
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline  # Sentiment analysis models
import os
import sys
import subprocess
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from keybert import KeyBERT
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline, AutoTokenizer
import re

huggingface_bin_path = "/root/anaconda3/lib/python3.11/site-packages"
os.environ["PATH"] = f"{huggingface_bin_path}:{os.environ['PATH']}"

subprocess.run(["huggingface-cli", "login", "--token", 'hf_rBIvOKVBOpXHAokVyWPusQygdsVdThA134'])

def fetch_top_100_cryptos():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    top_cryptos = pd.DataFrame({
        "name": [coin["name"] for coin in data],
        "symbol": [coin["symbol"] for coin in data]
    })

    return top_cryptos


def preprocess_text(text):
    cleaned_text = re.sub(r"ÃƒÂ¨Ã¢â‚¬Å¾Ã¢â€žÂ¢ÃƒÂ¨Ã…â€™ ÃƒÂ¨Ã… Ã¢â‚¬â„¢ÃƒÂ©Ã‹â€ Ã‚Â§ÃƒÂ®Ã†â€™ ÃƒÂ¤Ã‚Â¹Ã¢â‚¬Â¦ÃƒÂ¥Ã¢â‚¬Â¦Ã‚Â¤ÃƒÂ¦Ã‚ÂÃ¢â‚¬â€ÃƒÂ¥Ã‚ÂÃ‚Â®ÃƒÂ§Ã‚Â£Ã¢â‚¬â€", "", text)  # Removing bullet point character
    cleaned_text = re.sub(r"[^A-Za-z0-9\s,.]", "", cleaned_text)  # Remove special characters
    return cleaned_text

def preprocess_content_column(df):
    df['content'] = df['content'].apply(preprocess_text)
    return df

def fetch_crypto_data(keyword):
    """
    Fetch historical price data for the extracted keyword (cryptocurrency).
    Fetches the last 3 months of daily prices and trading volumes.
    """
    try:
        # Fetch cryptocurrency info based on keyword (symbol or name)
        coin_data = cg.get_price(ids=keyword, vs_currencies='usd', include_market_cap=True)
        market_data = cg.get_coin_market_chart_by_id(id=keyword, vs_currency='usd', days='90')
        return market_data['prices'], market_data['total_volumes']
    except Exception as e:
        print(f"Error fetching data for {keyword}: {e}")
        return None, None

def fetch_price_and_volume_data(symbol, days=60):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()

    if 'prices' in data and 'total_volumes' in data:
        prices = [x[1] for x in data['prices']]  # Extract prices
        volumes = [x[1] for x in data['total_volumes']]  # Extract volumes
        return prices, volumes
    else:
        print(f"Data not found for {symbol}")
        return None, None

# Step 1: Define TCN Model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, dropout=dropout
                )
            ]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        out = self.network(x.transpose(1, 2))
        out = out[:, :, -1]
        return self.fc(out)

# Step 2: AR-RNN Model
class ARRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(ARRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(data, seq_length=30):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.squeeze(), dtype=torch.float32), scaler

# Training function
def train_model(model, X, y, epochs=25):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model

# Prediction function
def predict(model, X):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    return predictions

def predict_price_and_volume(symbol, seq_length=30, epochs=25):
    prices, volumes = fetch_price_and_volume_data(symbol)
    if prices is None or volumes is None:
        return None, None

    X_price, y_price, scaler_price = prepare_data(prices, seq_length)
    X_volume, y_volume, scaler_volume = prepare_data(volumes, seq_length)

    # Initialize models
    tcn_model_price = TCN(input_size=1, output_size=1, num_channels=[25, 25, 25])
    tcn_model_volume = TCN(input_size=1, output_size=1, num_channels=[25, 25, 25])
    arrnn_model_price = ARRNN(input_size=1, hidden_size=50, output_size=1)
    arrnn_model_volume = ARRNN(input_size=1, hidden_size=50, output_size=1)

    # Train models
    tcn_model_price = train_model(tcn_model_price, X_price, y_price, epochs)
    arrnn_model_price = train_model(arrnn_model_price, X_price, y_price, epochs)
    tcn_model_volume = train_model(tcn_model_volume, X_volume, y_volume, epochs)
    arrnn_model_volume = train_model(arrnn_model_volume, X_volume, y_volume, epochs)

    # Make predictions
    price_pred_tcn = predict(tcn_model_price, X_price[-1].unsqueeze(0))
    price_pred_arrnn = predict(arrnn_model_price, X_price[-1].unsqueeze(0))
    volume_pred_tcn = predict(tcn_model_volume, X_volume[-1].unsqueeze(0))
    volume_pred_arrnn = predict(arrnn_model_volume, X_volume[-1].unsqueeze(0))

    # Average predictions
    avg_price_pred = (price_pred_tcn + price_pred_arrnn) / 2
    avg_volume_pred = (volume_pred_tcn + volume_pred_arrnn) / 2

    # Inverse scale the predictions
    avg_price_pred = scaler_price.inverse_transform(avg_price_pred)
    avg_volume_pred = scaler_volume.inverse_transform(avg_volume_pred)

    return avg_price_pred[0][0], avg_volume_pred[0][0]


finbert_pipeline = pipeline("text-classification", model="yiyanghkust/finbert-tone")

def sentiment_analysis(content):
    if isinstance(content, str):
        content = content[:512]
        finbert_result = finbert_pipeline(content)
        return finbert_result
    else:
        raise ValueError(f"Expected input of type 'str', but got {type(content)}")


telegram_token = '6467788089:AAEtl6sWveTLIP_cgJdliAS6l3wwUqXqrj8'  
en_user_id = '-1602149719671'  

def send_message_to_user(message, flag='en'):
    max_message_length = 4096 
    message_chunks = [message[i:i + max_message_length] for i in range(0, len(message), max_message_length)]
    
    if flag == 'en':
        chat_id = en_user_id
    elif flag == 'per': 
        chat_id = "-2002231326723"
    else:
        return "Invalid flag"

    for chunk in message_chunks:
        url = f'https://api.telegram.org/bot{telegram_token}/sendMessage'
        payload = {
            'chat_id': chat_id,
            'text': chunk
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            return f"Failed to send message. Status code: {response.status_code}, Response content: {response.content}"
    
    return "Message(s) sent successfully!"
   

def send_and_print_message(message,flag='en'):
    result = send_message_to_user(message,flag=flag)
    print(message)
    print(result)

def extract_keywords(text,top_num=7):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    kw_model = KeyBERT(model=model_name)

    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_num)
    
    return keywords   
    

def translate_to_persian(text, max_length=512):
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    

    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    tokenizer.src_lang = "en_XX"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)

    if "fa_IR" not in tokenizer.lang_code_to_id:
        raise ValueError("Farsi language code 'fa_IR' not found in tokenizer!")
    

    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fa_IR"])

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return translation[0]  # Return the first (and only) translation


def summarize_text_with_bart(text, max_length=1024, min_length=50, do_sample=False):
    summarization_pipeline = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if torch.cuda.is_available() else -1)  # Enable CUDA
    try:
        if len(text.split()) > 4096:
            text = " ".join(text.split()[:4096])  # Truncate to 4096 tokens if too long

        summary = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text
    

def process_dataframe(df, source):
   
    try:
        checklist_df = pd.read_csv('/root/checklist.csv')
    except FileNotFoundError:
        checklist_df = pd.DataFrame(columns=['title'])
    
    for index, row in df.iterrows():
        title = row['title']
        summary = row['summary']
        url = row['url'] if 'url' in row else ''
        
        if title in checklist_df['title'].values:
            print(f"-> Duplicate news: {title}")
            continue
        
        sentiment_result = sentiment_analysis(summary)
        sentiment_score = sentiment_result['score']
        sentiment_label = sentiment_result['label']
        sentiment = f"ðŸŸ¢ Positive âž¡ï¸ (score = {sentiment_score * 100:.1f}%)" if sentiment_label == 'positive' else f"ðŸ”´ Negative âž¡ï¸ (score = {sentiment_score * 100:.1f}%)"
        
        translated_summary = translate_to_persian(summary)
        
        message = (
            f"ðŸ“‹ Ø¹Ù†ÙˆØ§Ù†: {title}\n"
            f"ðŸš¨ Ø®Ù„Ø§ØµÙ‡ Ø®Ø¨Ø±: {translated_summary}\n"
            f"ØªØ­Ù„ÛŒÙ„ Ø§Ø­Ø³Ø§Ø³: {sentiment}\n"
            f"[Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ø¨ÛŒØ´ØªØ±]({url})\n"
            "Channel: @crypto_sentiment"
        )
        
        send_and_print_message(message, flag='per')

        checklist_df = pd.concat([checklist_df, pd.DataFrame([{'title': title}])], ignore_index=True)
        checklist_df.to_csv('/root/checklist.csv', index=False, encoding='utf-8')

        print(f"-> News sent: {title}")
        

def summarize_text(text, max_length=4096, min_length=30, do_sample=False):
    summarization_pipeline = pipeline('summarization', model='google/pegasus-large' )
    
    try:
        if len(text.split()) > 4096:
            text = " ".join(text.split()[:4096])  # Truncate text to 4096 tokens if too long

        summary = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']
    
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text

def drop_short_content(df):
    return df[df['content'].str.len() >= 60]

from datetime import datetime, timedelta


if __name__ == "__main__":
    top_cryptos = fetch_top_100_cryptos()
    coinmarketcap_df = pd.read_csv("coinmarketcap_articles.csv")
    coinmarketcap_df = preprocess_content_column(coinmarketcap_df)

    cg = CoinGeckoAPI()

    currency_symbols = ['bitcoin', 'ethereum', 'litecoin']  # Example list of symbols

    for symbol in currency_symbols:
        avg_price, avg_volume = predict_price_and_volume(symbol)
        if avg_price and avg_volume:
            sentiment_text = f"The future outlook for {symbol} is promising with a strong market presence."
            sentiment = sentiment_analysis(sentiment_text)
            print(f" Symbol: {symbol}, Avg Price Prediction: {avg_price}, Avg Volume Prediction: {avg_volume}, Sentiment: {sentiment}")
            message = f" Symbol: {symbol}, Avg Price Prediction: {avg_price}, Avg Volume Prediction: {avg_volume}, Sentiment: {sentiment}"

    # Loading your CSV files
    coinmarketcap_df = pd.read_csv("coinmarketcap_articles.csv")
    cointelegraph_df = pd.read_csv("/root/CryptoNewsCrawler/articles.csv")

    cointelegraph_df = drop_short_content(cointelegraph_df)
    coinmarketcap_df = drop_short_content(coinmarketcap_df)
    
    current_date = datetime.now().date()
    yesterday_date = current_date - timedelta(days=1)

    cointelegraph_df['date'] = pd.to_datetime(cointelegraph_df['date'])
    coinmarketcap_df['date'] = pd.to_datetime(coinmarketcap_df['date'])

    cointelegraph_df = cointelegraph_df[(cointelegraph_df['date'].dt.date == current_date) | 
                                    (cointelegraph_df['date'].dt.date == yesterday_date)]

    coinmarketcap_df = coinmarketcap_df[(coinmarketcap_df['date'].dt.date == current_date) | 
                                    (coinmarketcap_df['date'].dt.date == yesterday_date)]


    cointelegraph_df['summary'] = cointelegraph_df['content'].head(20).apply(lambda x: summarize_text_with_bart(str(x)))
    coinmarketcap_df['summary'] = coinmarketcap_df['content'].head(20).apply(lambda x: summarize_text_with_bart(str(x)))

    process_dataframe(cointelegraph_df.head(20),"cointelegraph")
    process_dataframe(coinmarketcap_df.head(20),"coinmarketcap")

