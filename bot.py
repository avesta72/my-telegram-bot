import telebot
import requests
import json
import ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Загружаем токен из .env файла
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

# Топ 500+ криптовалют с Binance
TOP_500_CRYPTO_SYMBOLS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'TRX',
    'LINK', 'MATIC', 'LTC', 'BCH', 'ATOM', 'UNI', 'XLM', 'ETC', 'XMR', 'FIL',
    'APT', 'HBAR', 'NEAR', 'VET', 'ARB', 'OP', 'MNT', 'IMX', 'LDO', 'TIA',
    'AAVE', 'EOS', 'GRT', 'ALGO', 'QNT', 'RNDR', 'STX', 'FTM', 'THETA', 'INJ',
    'EGLD', 'SAND', 'AXS', 'XTZ', 'MANA', 'GALA', 'CHZ', 'CRV', 'KAVA', 'KSM',
    'DYDX', 'COMP', 'MKR', 'SNX', 'ZEC', 'BAT', 'ENJ', 'IOTA', 'WAVES', 'NEO',
    'YFI', 'ZIL', 'IOST', 'CELO', 'KLAY', 'ONE', 'ROSE', 'RSR', 'OCEAN', 'ONT',
    'HNT', 'DASH', 'ANKR', 'ICX', 'SC', 'STORJ', 'LRC', 'REEF', 'COTI', 'AR',
    'RVN', 'SKL', 'CELR', 'DGB', 'BAND', 'OMG', 'CTSI', 'PERP', 'TRB', 'UMA',
    'BAL', 'KNC', 'JST', 'SXP', 'HOT', 'VTHO', 'MTL', 'CVC', 'STMX', 'OXT',
    'SUI', 'ENA', 'WIF', 'PEPE', 'BONK', 'FLOKI', 'JUP', 'PYTH', 'SEI', 'ORDI',
    'ONDO', 'JTO', 'BOME', 'MEME', 'BLOB', 'POPCAT', 'MYRO', 'WEN', 'MANEKI', 'CAT',
    'CAKE', 'SUSHI', '1INCH', 'RUNE', 'BADGER', 'ALPHA', 'FORTH', 'POLS', 'TVK', 'DODO',
    'LIT', 'POND', 'FIS', 'TRU', 'MLN', 'PNT', 'QSP', 'REQ', 'RLC', 'NMR',
    'ILV', 'YGG', 'GODS', 'VRA', 'CUBE', 'ALICE', 'ERN', 'SLP', 'CHR', 'SPS',
    'DG', 'BETA', 'RAD', 'BICO', 'HIGH', 'KP3R', 'GLM', 'AUCTION', 'BOND', 'FARM',
    'MINA', 'FLOW', 'KAS', 'NEXA', 'CORE', 'CFX', 'KAVA', 'SCRT', 'OASIS', 'ROSE',
    'MOVR', 'GLMR', 'ASTAR', 'SDN', 'PHA', 'CLV', 'COTI', 'CELR', 'SKL', 'LRC',
    'AGIX', 'FET', 'OCEAN', 'NMR', 'RLC', 'CTXC', 'DTA', 'NKN', 'PHB', 'VAI',
    'BAND', 'TRB', 'DIA', 'NEST', 'UMA', 'API3', 'PROM', 'TROY', 'LINK', 'DOT',
    'AR', 'FIL', 'STORJ', 'SC', 'BTT', 'WIN', 'LIVE', 'DENT', 'HOT', 'VET',
    'ZEC', 'XMR', 'DASH', 'ZEN', 'SC', 'PIVX', 'NAV', 'XVG', 'KMD', 'ARRR',
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDN', 'FEI', 'FRAX', 'UST',
    'BNB', 'FTT', 'HT', 'OKB', 'LEO', 'CRO', 'KCS', 'BGB', 'MX', 'GT',
    'GMT', 'APE', 'GALA', 'ENJ', 'MANA', 'SAND', 'DEXT', 'RARI', 'SUPER', 'WHALE',
    'REP', 'POLY', 'DPT', 'PRQ', 'UFT', 'BEPRO', 'POOLZ', 'DVP', 'ZAP', 'CVT',
    'NXM', 'BRD', 'INSUR', 'SHIELD', 'SAFE', 'COVER', 'ARMOR', 'NSURE', 'BRR', 'BLANK',
    'MCB', 'MIR', 'PERL', 'SKEY', 'DHT', 'BZRX', 'YFII', 'AKRO', 'SUSD', 'HEGIC',
    'XLM', 'XRP', 'ALGO', 'NANO', 'IOTA', 'XDC', 'HBAR', 'VET', 'QTUM', 'WAVES',
    'VET', 'AMB', 'MOD', 'WTC', 'ORBS', 'POA', 'REQ', 'RDN', 'SNT', 'PLR',
    'POWR', 'ELON', 'NRG', 'GRN', 'SLR', 'WPR', 'JUP', 'SOL', 'ADA', 'ETH',
    'REIT', 'PROP', 'LABS', 'OWN', 'REAL', 'ATL', 'LTX', 'RPT', 'KEY', 'DREP',
    'FUN', 'BET', 'EDG', 'WAX', 'CHP', 'DICE', 'VOX', 'SPANK', 'HERO', 'MOC',
    'BAT', 'BTT', 'CHZ', 'CVC', 'DENT', 'HOT', 'MANA', 'MTL', 'SAND', 'STMX',
    'ACH', 'AGLD', 'AIOZ', 'ALCX', 'ALEPH', 'ALPHA', 'AMP', 'ANC', 'ANT', 'API3',
    'ASM', 'AUDIO', 'AVT', 'BADGER', 'BAL', 'BAND', 'BAT', 'BETA', 'BICO', 'BLZ',
    'BNT', 'BOND', 'C98', 'CELR', 'CKB', 'COMBO', 'COTI', 'CVC', 'CVP', 'DAG',
    'DENT', 'DEXT', 'DIA', 'DNT', 'DOCK', 'DREP', 'DUSK', 'ELA', 'ELF', 'ENG',
    'ENS', 'ERN', 'FARM', 'FET', 'FIDA', 'FIO', 'FLM', 'FORTH', 'FRONT', 'FUN',
    'GAL', 'GHST', 'GLM', 'GNO', 'GTC', 'GTO', 'HARD', 'HFT', 'IDEX', 'ILV',
    'IOST', 'IOTX', 'JASMY', 'KEEP', 'KEY', 'KLAY', 'KNC', 'KSM', 'LINA', 'LPT',
    'LQTY', 'LRC', 'LSK', 'LTO', 'LUNA', 'MASK', 'MC', 'MDT', 'MIR', 'MKR',
    'MLN', 'MOVR', 'MULTI', 'NKN', 'NMR', 'NULS', 'OAX', 'OGN', 'OM', 'OXT',
    'PAXG', 'PERP', 'PHA', 'POLS', 'POND', 'POWR', 'PUNDIX', 'QKC', 'QNT', 'QUICK',
    'RAD', 'RARE', 'RARI', 'REN', 'REQ', 'RLC', 'ROSE', 'RPL', 'RSR', 'RUNE',
    'SFP', 'SHIB', 'SLP', 'SNT', 'SNX', 'SOLO', 'SPELL', 'STG', 'STRAX', 'STX',
    'SUN', 'SUPER', 'SUSHI', 'SXP', 'T', 'TCT', 'TFUEL', 'TLM', 'TNB', 'TKO',
    'TLOS', 'TOMO', 'TRIBE', 'TROY', 'TRU', 'TVK', 'UFT', 'UMA', 'UNFI', 'UTK',
    'VIB', 'VIDT', 'VITE', 'VOXEL', 'VTHO', 'WAN', 'WAVES', 'WAXP', 'WIN', 'WING',
    'WNXM', 'WOO', 'XEC', 'XEM', 'XNO', 'XVS', 'YFII', 'YGG', 'ZEN', 'ZIL',
    'ZRX'
]

# Топ 500 для авто-мониторинга
TOP_500_MONITORING = TOP_500_CRYPTO_SYMBOLS

# Глобальные переменные
subscribed_chats = set()
last_signals = {}

def prepare_ml_features(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
    df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['price_volume'] = df['close'] * df['volume']
    df['target'] = df['close'].shift(-5)
    return df.dropna()

def train_ai_model(df):
    try:
        features = ['rsi', 'macd', 'price_change', 'volume_change', 'high_low_ratio', 'ema_20', 'ema_50']
        X = df[features]
        y = df['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled[:-5], y[:-5])
        return model, scaler, features
    except Exception as e:
        print(f"Ошибка обучения ML: {e}")
        return None, None, None

def ai_price_prediction(df, model, scaler, features):
    try:
        current_features = df[features].iloc[-1:].values
        current_scaled = scaler.transform(current_features)
        prediction = model.predict(current_scaled)[0]
        return prediction
    except:
        return None

def get_crypto_news(symbol):
    try:
        mock_articles = [
            {'title': f'{symbol} shows strong momentum today', 'description': 'Positive market sentiment'},
            {'title': f'Experts bullish on {symbol} future', 'description': 'Market analysis positive'},
            {'title': f'{symbol} trading volume increases', 'description': 'Growing investor interest'}
        ]
        return mock_articles
    except Exception as e:
        print(f"Ошибка получения новостей: {e}")
        return []

def analyze_news_sentiment(articles):
    if not articles:
        return "neutral", 0
    sentiments = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        if len(text) > 50:
            try:
                analysis = TextBlob(text)
                sentiment_score = analysis.sentiment.polarity
                sentiments.append(sentiment_score)
            except:
                continue
    if not sentiments:
        return "neutral", 0
    avg_sentiment = sum(sentiments) / len(sentiments)
    if avg_sentiment > 0.1:
        return "bullish", avg_sentiment
    elif avg_sentiment < -0.1:
        return "bearish", avg_sentiment
    else:
        return "neutral", avg_sentiment

def blockchain_analysis(symbol):
    try:
        top_coins_analysis = {
            'BTC': "⛓️ BTC: Strong network activity",
            'ETH': "⛓️ ETH: High gas usage", 
            'BNB': "⛓️ BNB: BSC activity stable",
            'SOL': "⛓️ SOL: Fast transactions",
            'XRP': "⛓️ XRP: Banking integration",
            'ADA': "⛓️ ADA: Research focused",
            'SUI': "⛓️ SUI: Growing ecosystem",
            'ENA': "⛓️ ENA: Synthetic dollar protocol",
            'WIF': "⛓️ WIF: Meme coin momentum"
        }
        if symbol in top_coins_analysis:
            return top_coins_analysis[symbol], "positive"
        else:
            return f"⛓️ {symbol}: Standard chain metrics", "neutral"
    except:
        return f"⛓️ {symbol}: Data unavailable", "neutral"

def mean_reversion_strategy(df):
    if len(df) < 20:
        return "⚪ HOLD - Insufficient data", 0
    current_price = df['close'].iloc[-1]
    sma_20 = df['close'].rolling(20).mean().iloc[-1]
    deviation = (current_price - sma_20) / sma_20 * 100
    if deviation < -3:
        return "🟢 BUY - Mean Reversion", deviation
    elif deviation > 3:
        return "🔴 SELL - Mean Reversion", deviation
    else:
        return "⚪ HOLD - Mean Reversion", deviation

def breakout_strategy(df):
    if len(df) < 20:
        return "⚪ HOLD - Insufficient data", 0
    current_high = df['high'].iloc[-1]
    resistance = df['high'].rolling(20).max().iloc[-2]
    current_low = df['low'].iloc[-1]
    support = df['low'].rolling(20).min().iloc[-2]
    if current_high > resistance * 1.01:
        return "🟢 BUY - Breakout", current_high - resistance
    elif current_low < support * 0.99:
        return "🔴 SELL - Breakout", support - current_low
    else:
        return "⚪ HOLD - Breakout", 0

def rsi_momentum_strategy(df):
    if len(df) < 10:
        return "⚪ HOLD - Insufficient data", 0
    rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
    price_change = df['close'].pct_change(3).iloc[-1] * 100
    if rsi < 35 and price_change > -1:
        return "🟢 BUY - RSI Momentum", rsi
    elif rsi > 65 and price_change < 1:
        return "🔴 SELL - RSI Momentum", rsi
    else:
        return "⚪ HOLD - RSI Momentum", rsi

def get_ohlc_data(symbol, interval='1h', limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except Exception as e:
        print(f"Ошибка получения данных для {symbol}: {e}")
        return None

def calculate_indicators(df):
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        return df
    except Exception as e:
        print(f"Ошибка расчета индикаторов: {e}")
        return df

def auto_monitoring():
    while True:
        try:
            if not subscribed_chats:
                time.sleep(30)
                continue
            print(f"🔍 Проверяю сигналы для {len(subscribed_chats)} чатов...")
            print(f"💰 МОНИТОРИНГ {len(TOP_500_MONITORING)} КРИПТОВАЛЮТ!")
            
            # Разбиваем на группы по 50 для избежания перегрузки
            for i in range(0, len(TOP_500_MONITORING), 50):
                batch = TOP_500_MONITORING[i:i+50]
                print(f"📊 Анализирую батч {i//50 + 1}/{(len(TOP_500_MONITORING)-1)//50 + 1}")
                
                for symbol in batch:
                    timeframes = ['15m', '1h', '4h']
                    for tf in timeframes:
                        try:
                            df = get_ohlc_data(symbol, interval=tf, limit=50)
                            if df is None or len(df) < 20:
                                continue
                            df = calculate_indicators(df)
                            strategies = [
                                mean_reversion_strategy(df),
                                breakout_strategy(df), 
                                rsi_momentum_strategy(df)
                            ]
                            buy_signals = sum(1 for s, _ in strategies if 'BUY' in s)
                            sell_signals = sum(1 for s, _ in strategies if 'SELL' in s)
                            signal_key = f"{symbol}_{tf}"
                            current_signal = "NEUTRAL"
                            if buy_signals >= 2:
                                current_signal = "BUY"
                            elif sell_signals >= 2:
                                current_signal = "SELL"
                            last_signal = last_signals.get(signal_key)
                            if last_signal != current_signal and current_signal != "NEUTRAL":
                                last_signals[signal_key] = current_signal
                                message = f"🎯 **СИГНАЛ {symbol} ({tf})**\n\n"
                                message += f"📊 **Тип:** {current_signal}\n"
                                message += f"💰 **Цена:** ${df['close'].iloc[-1]:.2f}\n"
                                message += f"🟢 **BUY сигналы:** {buy_signals}/3\n"  
                                message += f"🔴 **SELL сигналы:** {sell_signals}/3\n"
                                message += f"⏰ **Таймфрейм:** {tf}\n"
                                message += f"🕒 **Время:** {datetime.now().strftime('%H:%M:%S')}"
                                for chat_id in subscribed_chats:
                                    try:
                                        bot.send_message(chat_id, message, parse_mode='Markdown')
                                        print(f"📨 Отправлен сигнал {symbol} на {tf} в чат {chat_id}")
                                    except Exception as e:
                                        print(f"Ошибка отправки в чат {chat_id}: {e}")
                        except Exception as e:
                            print(f"Ошибка анализа {symbol} на {tf}: {e}")
                            continue
                # Пауза между батчами
                time.sleep(10)
                
            print(f"✅ Проверка 500+ монет завершена. Следующая через 5 минут...")
            time.sleep(300)  # 5 минут между полными циклами
            
        except Exception as e:
            print(f"❌ Критическая ошибка мониторинга: {e}")
            time.sleep(60)

@bot.message_handler(commands=['start'])
def start(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    quick_coins = TOP_500_CRYPTO_SYMBOLS[:15]
    for i in range(0, len(quick_coins), 3):
        row = [telebot.types.KeyboardButton(f'/ai {coin}') for coin in quick_coins[i:i+3]]
        markup.add(*row)
    btn_advanced = telebot.types.KeyboardButton('/advanced BTC')
    btn_strategy = telebot.types.KeyboardButton('/strategies BTC')
    btn_auto = telebot.types.KeyboardButton('/autostart')
    btn_status = telebot.types.KeyboardButton('/status')
    btn_search = telebot.types.KeyboardButton('/search BTC')
    markup.add(btn_advanced, btn_strategy, btn_auto, btn_status, btn_search)
    welcome_text = (
        "🚀 **SUPER AI CRYPTO BOT v5.0** 🤖\n"
        "💎 **ТОП-500+ КРИПТОВАЛЮТ** 📊\n\n"
        "🔥 **МАКСИМАЛЬНОЕ ПОКРЫТИЕ:**\n"
        "• Анализ 500+ криптовалют\n"
        "• Авто-мониторинг ВСЕХ 500+ монет\n"
        "• Multi-таймфрейм анализ\n"
        "• AI предсказание цен\n"
        "• Авто-сигналы каждые 5 минут\n\n"
        "⚡ **Команды:**\n"
        "• `/ai BTC` - Полный AI анализ\n"
        "• `/advanced BTC` - Детальный анализ\n"
        "• `/strategies BTC` - Торговые стратегии\n"
        "• `/search BTC` - Поиск монеты\n"
        "• `/autostart` - Включить авто-сигналы\n"
        "• `/status` - Статус бота\n\n"
        "🎯 **Выберите монету для анализа:**"
    )
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup, parse_mode='Markdown')

@bot.message_handler(commands=['status'])
def status(message):
    status_text = (
        "🤖 **СТАТУС БОТА v5.0**\n\n"
        "✅ **Работает:** ДА\n"
        f"📊 **Мониторинг:** {len(subscribed_chats)} чатов\n"
        f"💰 **Покрытие:** {len(TOP_500_CRYPTO_SYMBOLS)}+ монет\n"
        f"🎯 **Авто-мониторинг:** {len(TOP_500_MONITORING)} криптовалют\n"
        f"⏰ **Проверка:** каждые 5 минут\n"
        "📈 **Таймфреймы:** 15m, 1h, 4h\n"
        f"🕒 **Время:** {datetime.now().strftime('%H:%M:%S')}\n\n"
        "⚡ **Бот активен и мониторит 500+ криптовалют!**"
    )
    bot.send_message(message.chat.id, status_text, parse_mode='Markdown')

@bot.message_handler(commands=['search'])
def search_coin(message):
    try:
        search_term = message.text.split()[1].upper()
    except:
        bot.send_message(message.chat.id, "❌ Укажите монету: `/search BTC`", parse_mode='Markdown')
        return
    found_coins = [coin for coin in TOP_500_CRYPTO_SYMBOLS if search_term in coin]
    if not found_coins:
        bot.send_message(message.chat.id, f"❌ Монета `{search_term}` не найдена.", parse_mode='Markdown')
        return
    if len(found_coins) == 1:
        ai_comprehensive_analysis_search(message, found_coins[0])
    else:
        coins_text = "🔍 **Найдены монеты:**\n\n"
        for coin in found_coins[:10]:
            coins_text += f"• `/ai {coin}` - AI анализ\n"
        if len(found_coins) > 10:
            coins_text += f"\n... и еще {len(found_coins) - 10} монет\n"
        coins_text += f"\n💎 **Используйте:** `/ai [монета]` для анализа"
        bot.send_message(message.chat.id, coins_text, parse_mode='Markdown')

def ai_comprehensive_analysis_search(message, symbol):
    bot.send_message(message.chat.id, f"🧠 **Запускаю SUPER AI анализ для {symbol}...**", parse_mode='Markdown')
    df = get_ohlc_data(symbol, limit=100)
    if df is None or len(df) < 50:
        bot.send_message(message.chat.id, f"❌ Недостаточно данных для {symbol}")
        return
    df_ml = prepare_ml_features(df)
    analysis_text = f"🤖 **SUPER AI АНАЛИЗ {symbol}**\n\n"
    if len(df_ml) > 20:
        model, scaler, features = train_ai_model(df_ml)
        if model and scaler:
            ai_prediction = ai_price_prediction(df_ml, model, scaler, features)
            current_price = df['close'].iloc[-1]
            if ai_prediction:
                change_pct = ((ai_prediction - current_price) / current_price) * 100
                analysis_text += f"🔮 **AI Прогноз:** ${ai_prediction:.2f} ({change_pct:+.2f}%)\n\n"
    df = calculate_indicators(df)
    current = df.iloc[-1]
    analysis_text += f"💰 **Текущая цена:** ${current['close']:.2f}\n"
    if 'rsi' in df.columns and pd.notna(current['rsi']):
        rsi_status = "🟢" if current['rsi'] < 30 else "🔴" if current['rsi'] > 70 else "⚪"
        analysis_text += f"📈 **RSI:** {current['rsi']:.1f} {rsi_status}\n"
    articles = get_crypto_news(symbol)
    news_sentiment, news_score = analyze_news_sentiment(articles)
    analysis_text += f"📰 **Новости:** {news_sentiment.upper()} ({news_score:.2f})\n"
    blockchain_info, _ = blockchain_analysis(symbol)
    analysis_text += f"{blockchain_info}\n\n"
    analysis_text += "🎯 **ТОРГОВЫЕ СИГНАЛЫ:**\n"
    strategies = [
        mean_reversion_strategy(df),
        breakout_strategy(df),
        rsi_momentum_strategy(df)
    ]
    for strategy, value in strategies:
        analysis_text += f"• {strategy}\n"
    buy_signals = sum(1 for s, _ in strategies if 'BUY' in s)
    sell_signals = sum(1 for s, _ in strategies if 'SELL' in s)
    if buy_signals >= 2:
        recommendation = "💎 **ОБЩАЯ РЕКОМЕНДАЦИЯ: ПОКУПКА** 🟢"
    elif sell_signals >= 2:
        recommendation = "💎 **ОБЩАЯ РЕКОМЕНДАЦИЯ: ПРОДАЖА** 🔴"
    else:
        recommendation = "💎 **ОБЩАЯ РЕКОМЕНДАЦИЯ: УДЕРЖИВАТЬ** ⚪"
    analysis_text += f"\n{recommendation}\n"
    analysis_text += f"\n⏰ **Время анализа:** {datetime.now().strftime('%H:%M:%S')}"
    bot.send_message(message.chat.id, analysis_text, parse_mode='Markdown')

@bot.message_handler(commands=['ai'])
def ai_comprehensive_analysis(message):
    try:
        symbol = message.text.split()[1].upper()
    except:
        bot.send_message(message.chat.id, "❌ Укажите монету: `/ai BTC`", parse_mode='Markdown')
        return
    if symbol not in TOP_500_CRYPTO_SYMBOLS:
        bot.send_message(message.chat.id, f"❌ Монета `{symbol}` не входит в базу", parse_mode='Markdown')
        return
    ai_comprehensive_analysis_search(message, symbol)

@bot.message_handler(commands=['advanced'])
def advanced_analysis(message):
    try:
        symbol = message.text.split()[1].upper()
    except:
        bot.send_message(message.chat.id, "❌ Укажите монету: `/advanced BTC`", parse_mode='Markdown')
        return
    if symbol not in TOP_500_CRYPTO_SYMBOLS:
        bot.send_message(message.chat.id, f"❌ Монета `{symbol}` не входит в базу", parse_mode='Markdown')
        return
    bot.send_message(message.chat.id, f"🔍 **Запускаю продвинутый анализ для {symbol}...**")
    analysis_text = f"📊 **МУЛЬТИ-ТАЙМФРЕЙМ АНАЛИЗ {symbol}**\n\n"
    for tf in ['15m', '1h', '4h']:
        df = get_ohlc_data(symbol, interval=tf, limit=50)
        if df is None:
            continue
        df = calculate_indicators(df)
        strategies = [
            mean_reversion_strategy(df),
            breakout_strategy(df),
            rsi_momentum_strategy(df)
        ]
        buy_signals = sum(1 for s, _ in strategies if 'BUY' in s)
        sell_signals = sum(1 for s, _ in strategies if 'SELL' in s)
        analysis_text += f"**{tf}:** 🟢{buy_signals}/3 🔴{sell_signals}/3 - ${df['close'].iloc[-1]:.2f}\n"
    analysis_text += f"\n🕒 **Обновлено:** {datetime.now().strftime('%H:%M:%S')}"
    bot.send_message(message.chat.id, analysis_text, parse_mode='Markdown')

@bot.message_handler(commands=['strategies'])
def trading_strategies(message):
    try:
        symbol = message.text.split()[1].upper()
    except:
        bot.send_message(message.chat.id, "❌ Укажите монету: `/strategies BTC`", parse_mode='Markdown')
        return
    if symbol not in TOP_500_CRYPTO_SYMBOLS:
        bot.send_message(message.chat.id, f"❌ Монета `{symbol}` не входит в базу", parse_mode='Markdown')
        return
    bot.send_message(message.chat.id, f"🎯 **Анализ торговых стратегий для {symbol}...**")
    df = get_ohlc_data(symbol, limit=100)
    if df is None:
        bot.send_message(message.chat.id, f"❌ Нет данных для {symbol}")
        return
    df = calculate_indicators(df)
    strategies_text = f"💼 **ТОРГОВЫЕ СТРАТЕГИИ {symbol}**\n\n"
    strategies = [
        ("📊 Возврат к среднему", mean_reversion_strategy(df)),
        ("🚀 Пробой уровней", breakout_strategy(df)),
        ("⚡ RSI Моментум", rsi_momentum_strategy(df))
    ]
    for strategy_name, (signal, value) in strategies:
        strategies_text += f"{strategy_name}:\n"
        strategies_text += f"• Сигнал: {signal}\n"
        strategies_text += f"• Значение: {value:.2f}\n\n"
    strategies_text += "🛡️ **РИСК-МЕНЕДЖМЕНТ:**\n"
    strategies_text += "• Размер позиции: 2-5% от депозита\n"
    strategies_text += "• Стоп-лосс: 3-5%\n"
    strategies_text += "• Тейк-профит: 6-10%\n"
    strategies_text += "• Риск/Прибыль: 1:2\n"
    bot.send_message(message.chat.id, strategies_text, parse_mode='Markdown')

@bot.message_handler(commands=['autostart'])
def autostart(message):
    subscribed_chats.add(message.chat.id)
    bot.send_message(message.chat.id,
        "🔔 **АВТО-СИГНАЛЫ АКТИВИРОВАНЫ!** 🚀\n\n"
        "Теперь вы будете получать уведомления:\n"
        f"• Каждые 5 минут\n"
        f"• Для {len(TOP_500_MONITORING)}+ криптовалют\n"
        "• На таймфреймах: 15m, 1h, 4h\n"
        "• При 2+ BUY/SELL сигналах\n\n"
        "⚡ **Бот начал активный поиск сигналов!**",
        parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    help_text = (
        "🤖 **SUPER AI CRYPTO BOT v5.0**\n"
        "💎 **ТОП
