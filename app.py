
import requests
import numpy as np
import time
import json
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

BOT_TOKEN = "7663938011:AAEHhNQI5NqgJLuW0NyxRzYwKmeg08kVTwU"
CHAT_ID = "6514168807"
SYMBOL = "BTCUSDT"
HISTORY_FILE = 'prediction_history.json'

class AdvancedTradingBot:
    def __init__(self):
        self.prediction_history = self.load_history()
        self.last_update_id = 0
        self.is_running = True
        self.data_collector_active = True
        self.market_data_cache = []
        self.ml_model_weights = self.load_ml_weights()
        self.prediction_accuracy_tracker = []
        self.continuous_learning_data = []
        
        # Start background data collection
        self.start_data_collection_thread()

    def load_history(self):
        """Tải lịch sử dự đoán từ file"""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []

    def save_history(self):
        """Lưu lịch sử dự đoán vào file"""
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.prediction_history[-1000:], f)
    
    def load_ml_weights(self):
        """Tải trọng số ML từ file"""
        weights_file = 'ml_weights.json'
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                return json.load(f)
        return {
            'rsi_weight': 0.25,
            'macd_weight': 0.20,
            'bb_weight': 0.15,
            'volume_weight': 0.15,
            'pattern_weight': 0.10,
            'divergence_weight': 0.15
        }
    
    def save_ml_weights(self):
        """Lưu trọng số ML"""
        with open('ml_weights.json', 'w') as f:
            json.dump(self.ml_model_weights, f)
    
    def start_data_collection_thread(self):
        """Khởi động thread thu thập dữ liệu liên tục"""
        def collect_data():
            while self.data_collector_active:
                try:
                    self.collect_market_data()
                    self.update_ml_model()
                    time.sleep(30)  # Thu thập mỗi 30 giây
                except Exception as e:
                    print(f"Data collection error: {e}")
                    time.sleep(60)
        
        data_thread = threading.Thread(target=collect_data, daemon=True)
        data_thread.start()
        print("📊 Background data collection started!")
    
    def collect_market_data(self):
        """Thu thập dữ liệu thị trường liên tục"""
        try:
            candles = self.get_candle_data(100)
            if not candles:
                return
            
            current_time = int(time.time() * 1000)
            indicators = self.calculate_advanced_indicators(candles)
            
            market_data = {
                'timestamp': current_time,
                'price': candles[-1]['close'],
                'volume': candles[-1]['volume'],
                'indicators': indicators,
                'market_structure': self.analyze_market_structure(candles),
                'patterns': self.detect_advanced_patterns(candles[-5:])
            }
            
            self.market_data_cache.append(market_data)
            
            # Giữ chỉ 1000 điểm dữ liệu gần nhất
            if len(self.market_data_cache) > 1000:
                self.market_data_cache = self.market_data_cache[-1000:]
                
            # Lưu dữ liệu định kỳ
            if len(self.market_data_cache) % 50 == 0:
                self.save_market_data()
                
        except Exception as e:
            print(f"Market data collection error: {e}")
    
    def save_market_data(self):
        """Lưu dữ liệu thị trường"""
        try:
            with open('market_data_cache.json', 'w') as f:
                json.dump(self.market_data_cache[-500:], f)
        except Exception as e:
            print(f"Save market data error: {e}")
    
    def detect_advanced_patterns(self, candles):
        """Phát hiện các pattern nâng cao"""
        if len(candles) < 5:
            return {'pattern': 'insufficient_data', 'strength': 0}
        
        patterns = []
        
        # Hammer pattern
        last = candles[-1]
        body = abs(last['close'] - last['open'])
        lower_shadow = last['open'] - last['low'] if last['close'] > last['open'] else last['close'] - last['low']
        upper_shadow = last['high'] - last['close'] if last['close'] > last['open'] else last['high'] - last['open']
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns.append({'name': 'hammer', 'strength': 70, 'direction': 'bullish'})
        
        # Shooting star
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns.append({'name': 'shooting_star', 'strength': 70, 'direction': 'bearish'})
        
        # Engulfing patterns
        if len(candles) >= 2:
            prev = candles[-2]
            curr = candles[-1]
            
            # Bullish engulfing
            if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                curr['open'] < prev['close'] and curr['close'] > prev['open']):
                patterns.append({'name': 'bullish_engulfing', 'strength': 80, 'direction': 'bullish'})
            
            # Bearish engulfing
            if (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                curr['open'] > prev['close'] and curr['close'] < prev['open']):
                patterns.append({'name': 'bearish_engulfing', 'strength': 80, 'direction': 'bearish'})
        
        if not patterns:
            return {'pattern': 'neutral', 'strength': 0}
        
        # Trả về pattern mạnh nhất
        strongest = max(patterns, key=lambda x: x['strength'])
        return strongest
    
    def update_ml_model(self):
        """Cập nhật mô hình ML dựa trên kết quả thực tế"""
        try:
            if len(self.market_data_cache) < 50:
                return
            
            # Kiểm tra accuracy của các dự đoán gần đây
            self.validate_recent_predictions()
            
            # Điều chỉnh trọng số dựa trên performance
            self.adjust_ml_weights()
            
            # Lưu trọng số mới
            self.save_ml_weights()
            
        except Exception as e:
            print(f"ML model update error: {e}")
    
    def validate_recent_predictions(self):
        """Xác thực các dự đoán gần đây"""
        current_time = int(time.time() * 1000)
        
        for prediction in self.prediction_history[-20:]:
            if 'validated' not in prediction:
                pred_time = prediction['timestamp']
                
                # Tìm dữ liệu sau 1-3 phút từ lúc dự đoán
                for data in self.market_data_cache:
                    if pred_time + 60000 <= data['timestamp'] <= pred_time + 180000:
                        actual_result = self.determine_actual_outcome(
                            prediction['price'], 
                            data['price']
                        )
                        
                        prediction['actual_result'] = actual_result
                        prediction['validated'] = True
                        prediction['accuracy'] = self.calculate_prediction_accuracy(
                            prediction, actual_result
                        )
                        
                        self.prediction_accuracy_tracker.append({
                            'timestamp': current_time,
                            'predicted': prediction.get('trend', 'unknown'),
                            'actual': actual_result,
                            'accuracy': prediction['accuracy']
                        })
                        break
    
    def determine_actual_outcome(self, pred_price, actual_price):
        """Xác định kết quả thực tế"""
        change_percent = ((actual_price - pred_price) / pred_price) * 100
        
        if change_percent > 0.1:
            return 'bullish'
        elif change_percent < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def calculate_prediction_accuracy(self, prediction, actual_result):
        """Tính độ chính xác của dự đoán"""
        predicted_trend = prediction.get('trend', '').lower()
        
        if 'tăng' in predicted_trend or 'mua' in predicted_trend:
            predicted = 'bullish'
        elif 'giảm' in predicted_trend or 'bán' in predicted_trend:
            predicted = 'bearish'
        else:
            predicted = 'neutral'
        
        if predicted == actual_result:
            return 100
        elif predicted == 'neutral' or actual_result == 'neutral':
            return 50
        else:
            return 0
    
    def adjust_ml_weights(self):
        """Điều chỉnh trọng số ML dựa trên performance"""
        if len(self.prediction_accuracy_tracker) < 10:
            return
        
        recent_accuracy = np.mean([
            acc['accuracy'] for acc in self.prediction_accuracy_tracker[-10:]
        ])
        
        # Nếu accuracy thấp, điều chỉnh trọng số
        if recent_accuracy < 75:
            # Tăng trọng số cho các chỉ số có performance tốt
            self.ml_model_weights['volume_weight'] *= 1.1
            self.ml_model_weights['pattern_weight'] *= 1.1
            
            # Giảm trọng số cho các chỉ số kém
            self.ml_model_weights['rsi_weight'] *= 0.95
            
        elif recent_accuracy > 85:
            # Tinh chỉnh nhẹ khi accuracy cao
            for key in self.ml_model_weights:
                self.ml_model_weights[key] *= 1.02
        
        # Normalize weights
        total_weight = sum(self.ml_model_weights.values())
        for key in self.ml_model_weights:
            self.ml_model_weights[key] /= total_weight

    def get_candle_data(self, limit=100):
        """Lấy dữ liệu nến với số lượng limit"""
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=1m&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            candles = []
            for candle in data:
                candles.append({
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            return candles
        except Exception as e:
            print(f"Error getting candle data: {e}")
            return []

    def calculate_advanced_indicators(self, candles):
        """Tính toán các chỉ số kỹ thuật nâng cao"""
        if len(candles) < 50:
            return {}

        prices = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        volumes = [c['volume'] for c in candles]

        indicators = {}

        # RSI cải tiến với nhiều timeframe
        indicators['rsi_14'] = self.calculate_rsi(prices, 14)
        indicators['rsi_21'] = self.calculate_rsi(prices, 21)
        indicators['rsi_7'] = self.calculate_rsi(prices, 7)

        # MACD với histogram
        macd, signal, histogram = self.calculate_macd_advanced(prices)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = histogram

        # Bollinger Bands với width
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(prices)
        indicators['bb_upper'] = upper_bb
        indicators['bb_middle'] = middle_bb
        indicators['bb_lower'] = lower_bb
        indicators['bb_width'] = (upper_bb - lower_bb) / middle_bb * 100
        indicators['bb_position'] = (prices[-1] - lower_bb) / (upper_bb - lower_bb) * 100

        # Stochastic Oscillator
        indicators['stoch_k'], indicators['stoch_d'] = self.calculate_stochastic(highs, lows, prices)

        # Williams %R
        indicators['williams_r'] = self.calculate_williams_r(highs, lows, prices)

        # EMA multiple timeframes
        indicators['ema_5'] = self.calculate_ema(prices, 5)
        indicators['ema_13'] = self.calculate_ema(prices, 13)
        indicators['ema_21'] = self.calculate_ema(prices, 21)
        indicators['ema_50'] = self.calculate_ema(prices, 50)

        # Volume indicators
        indicators['volume_sma'] = np.mean(volumes[-20:])
        indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
        indicators['volume_trend'] = self.analyze_volume_trend_advanced(volumes)

        # Support/Resistance levels
        indicators['support'], indicators['resistance'] = self.find_support_resistance(highs, lows, prices)

        # Price patterns
        indicators['pattern'] = self.detect_candlestick_patterns(candles[-3:])

        # Trend strength
        indicators['trend_strength'] = self.calculate_trend_strength(prices)

        return indicators

    def calculate_rsi(self, prices, period=14):
        """Tính RSI cải tiến"""
        if len(prices) < period + 1:
            return 50

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    def calculate_macd_advanced(self, prices):
        """Tính MACD với histogram"""
        if len(prices) < 26:
            return 0, 0, 0

        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        macd_line = ema12 - ema26

        # Tính signal line bằng EMA của MACD
        macd_values = []
        for i in range(26, len(prices)):
            ema12_i = self.calculate_ema(prices[:i + 1], 12)
            ema26_i = self.calculate_ema(prices[:i + 1], 26)
            macd_values.append(ema12_i - ema26_i)

        signal_line = self.calculate_ema(macd_values, 9) if len(macd_values) >= 9 else macd_line
        histogram = macd_line - signal_line

        return round(macd_line, 4), round(signal_line, 4), round(histogram, 4)

    def calculate_ema(self, prices, period):
        """Tính EMA"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Tính Bollinger Bands"""
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices)
        else:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        return upper_band, sma, lower_band

    def calculate_stochastic(self, highs, lows, closes, k_period=14, d_period=3):
        """Tính Stochastic Oscillator"""
        if len(highs) < k_period:
            return 50, 50

        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])

        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100

        # Tính %D (SMA của %K)
        k_values = []
        for i in range(k_period, len(closes) + 1):
            h_high = max(highs[i-k_period:i])
            l_low = min(lows[i-k_period:i])
            if h_high != l_low:
                k_val = ((closes[i-1] - l_low) / (h_high - l_low)) * 100
            else:
                k_val = 50
            k_values.append(k_val)

        d_percent = np.mean(k_values[-d_period:]) if len(k_values) >= d_period else k_percent

        return round(k_percent, 2), round(d_percent, 2)

    def calculate_williams_r(self, highs, lows, closes, period=14):
        """Tính Williams %R"""
        if len(highs) < period:
            return -50

        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])

        if highest_high == lowest_low:
            return -50

        williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
        return round(williams_r, 2)

    def analyze_volume_trend_advanced(self, volumes):
        """Phân tích xu hướng volume nâng cao"""
        if len(volumes) < 20:
            return 'neutral'

        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[-20:])

        if recent_volume > avg_volume * 1.5:
            return 'very_high'
        elif recent_volume > avg_volume * 1.2:
            return 'high'
        elif recent_volume < avg_volume * 0.7:
            return 'low'
        else:
            return 'normal'

    def find_support_resistance(self, highs, lows, prices):
        """Tìm mức support và resistance"""
        if len(prices) < 20:
            return prices[-1] * 0.98, prices[-1] * 1.02

        recent_prices = prices[-20:]
        support = min(recent_prices)
        resistance = max(recent_prices)

        return support, resistance

    def detect_candlestick_patterns(self, candles):
        """Phát hiện mô hình nến"""
        if len(candles) < 3:
            return "neutral"

        last = candles[-1]
        prev = candles[-2]

        body_size = abs(last['close'] - last['open'])
        candle_range = last['high'] - last['low']

        if body_size < candle_range * 0.3:
            return "doji"
        elif last['close'] > last['open'] and body_size > candle_range * 0.7:
            return "strong_bullish"
        elif last['close'] < last['open'] and body_size > candle_range * 0.7:
            return "strong_bearish"
        else:
            return "neutral"

    def calculate_trend_strength(self, prices):
        """Tính độ mạnh của xu hướng"""
        if len(prices) < 20:
            return 50

        short_ema = self.calculate_ema(prices, 10)
        long_ema = self.calculate_ema(prices, 20)

        trend_strength = abs((short_ema - long_ema) / long_ema * 100)
        return min(trend_strength * 10, 100)

    def advanced_ai_predict(self, candles):
        """AI dự đoán nâng cao với Smart Money Concepts và ML"""
        indicators = self.calculate_advanced_indicators(candles)
        current_price = candles[-1]['close']

        # Tính ATR cho risk management
        atr = self.calculate_atr(candles)

        # Smart Money Analysis
        market_structure = self.analyze_market_structure(candles)
        liquidity_zones = self.identify_liquidity_zones(candles)
        order_blocks = self.find_order_blocks(candles)
        fair_value_gaps = self.detect_fair_value_gaps(candles)

        # Advanced pattern analysis
        pattern_analysis = self.detect_advanced_patterns(candles[-5:])

        # Tính điểm tổng hợp với ML weights
        score = 0
        confidence_factors = []
        signals = []

        # RSI Analysis với dynamic weight
        rsi_weight = self.ml_model_weights['rsi_weight'] * 100
        if indicators['rsi_14'] < 30:
            score += rsi_weight
            confidence_factors.append("RSI oversold")
            signals.append("🟢 RSI quá bán")
        elif indicators['rsi_14'] > 70:
            score -= rsi_weight
            confidence_factors.append("RSI overbought")
            signals.append("🔴 RSI quá mua")

        # MACD Analysis với dynamic weight
        macd_weight = self.ml_model_weights['macd_weight'] * 100
        if indicators['macd_histogram'] > 0 and indicators['macd'] > indicators['macd_signal']:
            score += macd_weight
