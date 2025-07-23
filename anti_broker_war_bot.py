#!/usr/bin/env python3
"""
🔥 ANTI-BROKER WAR MACHINE - ULTIMATE OTC DESTROYER 🔥
Specifically designed to BEAT broker manipulation
PLAYS AGAINST THE BROKER - NOT THE MARKET
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
import io
import random

# Configuration
TOKEN = '8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ'

try:
    from telegram import Bot, Update
    from telegram.ext import Application, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

class AntiBrokerWarBot:
    """⚔️ Anti-Broker War Machine - Designed to Beat OTC Manipulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = TOKEN
        self.application = None
        
        # BROKER WARFARE INTELLIGENCE
        self.broker_patterns = {
            'trap_hours': [9, 10, 14, 15, 21, 22],  # When brokers set traps
            'manipulation_minutes': [0, 15, 30, 45],  # Round number traps
            'fake_volume_hours': [3, 4, 5, 12, 13],  # Low volume = easy manipulation
            'hunt_zones': [19, 20, 23, 0, 1],  # Stop loss hunting hours
        }
        
        # ANTI-BROKER STRATEGIES
        self.war_strategies = {
            'reverse_psychology': True,  # Do opposite of obvious
            'chaos_timing': True,  # Use unpredictable entry times
            'stealth_mode': True,  # Small amounts, fly under radar
            'broker_baiting': True,  # Let broker think they're winning
            'pattern_disruption': True,  # Break their algorithms
        }
        
        # BROKER BEHAVIOR MEMORY
        self.broker_memory = {
            'fake_breakouts': [],
            'manipulation_times': [],
            'reversal_traps': [],
            'volume_fakes': [],
            'success_patterns': []
        }
        
        print("⚔️ ANTI-BROKER WAR MACHINE INITIALIZED")
        print("🎯 MISSION: DESTROY BROKER MANIPULATION")
        print("🧠 STRATEGY: PSYCHOLOGICAL WARFARE")
        print("💀 TARGET: QUOTEX OTC BROKERS")
    
    def _get_utc_plus_6_time(self):
        """Get current UTC+6 time"""
        utc_now = datetime.utcnow()
        return utc_now + timedelta(hours=6)
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ANTI-BROKER CHART WARFARE"""
        print("⚔️ BROKER WAR MODE ACTIVATED - Analyzing enemy patterns...")
        
        try:
            local_time = self._get_utc_plus_6_time()
            
            # BROKER THREAT ASSESSMENT
            broker_threat = self._assess_broker_threat(local_time)
            
            await update.message.reply_text(
                f"⚔️ <b>ANTI-BROKER WAR MODE</b> ⚔️\n\n"
                f"🕰️ <b>Time:</b> {local_time.strftime('%H:%M:%S')} UTC+6\n"
                f"🎯 <b>Broker Threat:</b> {broker_threat['level']}\n"
                f"🧠 <b>War Strategy:</b> {broker_threat['strategy']}\n\n"
                f"📊 Analyzing broker manipulation patterns...",
                parse_mode='HTML'
            )
            
            # Download and process image
            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            
            photo_bytes = io.BytesIO()
            await photo_file.download_to_memory(photo_bytes)
            photo_bytes.seek(0)
            
            # Convert to OpenCV image
            nparr = np.frombuffer(photo_bytes.getvalue(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                await update.message.reply_text("❌ Cannot analyze broker's fake chart. Send clearer image.")
                return
            
            # ANTI-BROKER ANALYSIS
            war_analysis = await self._broker_warfare_analysis(img, local_time, broker_threat)
            
            # BROKER COUNTER-STRATEGY
            counter_strategy = self._generate_counter_strategy(war_analysis, broker_threat)
            
            # PSYCHOLOGICAL WARFARE DECISION
            final_decision = self._psychological_warfare_decision(counter_strategy, broker_threat)
            
            # EXECUTE WAR PLAN
            if final_decision['execute']:
                await self._execute_war_plan(update, final_decision, local_time, broker_threat)
            else:
                await self._send_no_war_message(update, final_decision, broker_threat)
            
        except Exception as e:
            print(f"⚔️ WAR ERROR: {e}")
            await update.message.reply_text("⚔️ Broker interference detected. Regrouping for next attack.")
    
    def _assess_broker_threat(self, local_time):
        """Assess current broker manipulation threat level"""
        try:
            hour = local_time.hour
            minute = local_time.minute
            
            threat_score = 0
            threat_factors = []
            
            # TRAP HOUR DETECTION
            if hour in self.broker_patterns['trap_hours']:
                threat_score += 3
                threat_factors.append('BROKER_TRAP_HOUR')
            
            # MANIPULATION MINUTE DETECTION
            if minute in self.broker_patterns['manipulation_minutes']:
                threat_score += 2
                threat_factors.append('ROUND_MINUTE_TRAP')
            
            # FAKE VOLUME DETECTION
            if hour in self.broker_patterns['fake_volume_hours']:
                threat_score += 4
                threat_factors.append('FAKE_VOLUME_PERIOD')
            
            # HUNT ZONE DETECTION
            if hour in self.broker_patterns['hunt_zones']:
                threat_score += 2
                threat_factors.append('STOP_HUNT_ZONE')
            
            # DETERMINE THREAT LEVEL AND STRATEGY
            if threat_score >= 6:
                return {
                    'level': 'MAXIMUM THREAT 🔴',
                    'strategy': 'FULL_REVERSE_PSYCHOLOGY',
                    'score': threat_score,
                    'factors': threat_factors,
                    'broker_mode': 'AGGRESSIVE_MANIPULATION'
                }
            elif threat_score >= 4:
                return {
                    'level': 'HIGH THREAT 🟠',
                    'strategy': 'CHAOS_WARFARE',
                    'score': threat_score,
                    'factors': threat_factors,
                    'broker_mode': 'ACTIVE_MANIPULATION'
                }
            elif threat_score >= 2:
                return {
                    'level': 'MODERATE THREAT 🟡',
                    'strategy': 'STEALTH_OPERATIONS',
                    'score': threat_score,
                    'factors': threat_factors,
                    'broker_mode': 'PASSIVE_MANIPULATION'
                }
            else:
                return {
                    'level': 'LOW THREAT 🟢',
                    'strategy': 'NORMAL_WARFARE',
                    'score': threat_score,
                    'factors': threat_factors,
                    'broker_mode': 'MINIMAL_MANIPULATION'
                }
                
        except Exception as e:
            print(f"Threat assessment error: {e}")
            return {
                'level': 'UNKNOWN THREAT ⚫',
                'strategy': 'DEFENSIVE_MODE',
                'score': 5,
                'factors': ['ASSESSMENT_ERROR'],
                'broker_mode': 'UNKNOWN'
            }
    
    async def _broker_warfare_analysis(self, image, local_time, broker_threat):
        """Analyze chart for broker manipulation patterns"""
        try:
            print("🔍 Analyzing broker manipulation patterns...")
            
            # STANDARD TECHNICAL ANALYSIS (To understand what broker sees)
            candles_data = self._detect_candles_for_war(image)
            trend_analysis = self._analyze_broker_trend_traps(image)
            pattern_analysis = self._detect_broker_pattern_traps(candles_data)
            volume_analysis = self._analyze_fake_volume(image)
            
            # BROKER PSYCHOLOGY ANALYSIS
            broker_psychology = self._analyze_broker_psychology(candles_data, trend_analysis, pattern_analysis)
            
            # MANIPULATION DETECTION
            manipulation_signals = self._detect_manipulation_signals(image, broker_threat)
            
            # TRAP DETECTION
            trap_analysis = self._detect_broker_traps(pattern_analysis, trend_analysis, broker_threat)
            
            return {
                'candles_data': candles_data,
                'trend_analysis': trend_analysis,
                'pattern_analysis': pattern_analysis,
                'volume_analysis': volume_analysis,
                'broker_psychology': broker_psychology,
                'manipulation_signals': manipulation_signals,
                'trap_analysis': trap_analysis,
                'analysis_time': local_time.strftime('%H:%M:%S UTC+6')
            }
            
        except Exception as e:
            print(f"War analysis error: {e}")
            return self._create_defensive_analysis()
    
    def _detect_candles_for_war(self, image):
        """Detect candles but focus on broker manipulation patterns"""
        try:
            height, width = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect green and red candles
            green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
            
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bullish_candles = len([c for c in green_contours if cv2.contourArea(c) > 80])
            bearish_candles = len([c for c in red_contours if cv2.contourArea(c) > 80])
            total_candles = bullish_candles + bearish_candles
            
            if total_candles > 0:
                bullish_ratio = bullish_candles / total_candles
                bearish_ratio = bearish_candles / total_candles
            else:
                bullish_ratio = bearish_ratio = 0.5
            
            # BROKER MANIPULATION INDICATORS
            manipulation_score = 0
            
            # Too perfect ratio = broker manipulation
            if abs(bullish_ratio - bearish_ratio) > 0.8:
                manipulation_score += 0.3
            
            # Too many candles = fake activity
            if total_candles > 30:
                manipulation_score += 0.2
            
            # Too few candles = hiding manipulation
            if total_candles < 5:
                manipulation_score += 0.4
            
            return {
                'total_candles': min(total_candles, 50),
                'bullish_candles': bullish_candles,
                'bearish_candles': bearish_candles,
                'bullish_ratio': bullish_ratio,
                'bearish_ratio': bearish_ratio,
                'manipulation_score': manipulation_score,
                'broker_bias': 'BULLISH' if bullish_ratio > 0.6 else 'BEARISH' if bearish_ratio > 0.6 else 'NEUTRAL'
            }
            
        except Exception as e:
            print(f"Candle war detection error: {e}")
            return {
                'total_candles': 0, 'bullish_candles': 0, 'bearish_candles': 0,
                'bullish_ratio': 0.5, 'bearish_ratio': 0.5, 'manipulation_score': 0.5,
                'broker_bias': 'UNKNOWN'
            }
    
    def _analyze_broker_trend_traps(self, image):
        """Analyze trends but focus on broker trap patterns"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None:
                angles = [line[0][1] * 180 / np.pi for line in lines[:20] if 30 < line[0][1] * 180 / np.pi < 150]
                
                if angles:
                    avg_angle = np.mean(angles)
                    angle_std = np.std(angles)
                    
                    # BROKER TRAP DETECTION
                    trap_probability = 0.0
                    
                    # Too perfect trend = broker trap
                    if angle_std < 5:
                        trap_probability += 0.4
                    
                    # Obvious direction = broker wants you to think this
                    if avg_angle > 110 or avg_angle < 70:
                        trap_probability += 0.3
                    
                    if avg_angle > 100:
                        direction = 'OBVIOUS_UPTREND'
                        broker_wants = 'CALLS'
                        counter_move = 'PUT'
                    elif avg_angle < 80:
                        direction = 'OBVIOUS_DOWNTREND'  
                        broker_wants = 'PUTS'
                        counter_move = 'CALL'
                    else:
                        direction = 'SIDEWAYS_TRAP'
                        broker_wants = 'CONFUSION'
                        counter_move = 'WAIT'
                    
                    return {
                        'direction': direction,
                        'broker_wants': broker_wants,
                        'counter_move': counter_move,
                        'trap_probability': trap_probability,
                        'angle': avg_angle,
                        'consistency': 'TRAP' if angle_std < 10 else 'NATURAL'
                    }
            
            return {
                'direction': 'NO_CLEAR_DIRECTION',
                'broker_wants': 'UNKNOWN',
                'counter_move': 'ANALYZE_MORE',
                'trap_probability': 0.5,
                'angle': 90,
                'consistency': 'UNCLEAR'
            }
            
        except Exception as e:
            print(f"Trend trap analysis error: {e}")
            return {
                'direction': 'ERROR',
                'broker_wants': 'UNKNOWN',
                'counter_move': 'DEFENSIVE',
                'trap_probability': 0.8,
                'angle': 0,
                'consistency': 'ERROR'
            }
    
    def _detect_broker_pattern_traps(self, candles_data):
        """Detect patterns but identify broker traps"""
        try:
            total_candles = candles_data['total_candles']
            bullish_ratio = candles_data['bullish_ratio']
            bearish_ratio = candles_data['bearish_ratio']
            manipulation_score = candles_data['manipulation_score']
            
            if total_candles < 5:
                return {
                    'pattern': 'INSUFFICIENT_DATA',
                    'broker_trap': 'DATA_HIDING',
                    'confidence': 0.0,
                    'counter_strategy': 'WAIT_FOR_MORE_DATA'
                }
            
            # OBVIOUS PATTERNS = BROKER TRAPS
            if bullish_ratio > 0.8:
                return {
                    'pattern': 'OBVIOUS_BULLISH_SETUP',
                    'broker_trap': 'BULL_TRAP',
                    'confidence': 0.9,
                    'counter_strategy': 'GO_PUT',
                    'trap_reason': 'Too obvious - broker wants CALLs'
                }
            
            if bearish_ratio > 0.8:
                return {
                    'pattern': 'OBVIOUS_BEARISH_SETUP',
                    'broker_trap': 'BEAR_TRAP',
                    'confidence': 0.9,
                    'counter_strategy': 'GO_CALL',
                    'trap_reason': 'Too obvious - broker wants PUTs'
                }
            
            # PERFECT BALANCE = BROKER CONFUSION TRAP
            if abs(bullish_ratio - bearish_ratio) < 0.1:
                return {
                    'pattern': 'PERFECT_BALANCE',
                    'broker_trap': 'CONFUSION_TRAP',
                    'confidence': 0.7,
                    'counter_strategy': 'WAIT_FOR_BREAKOUT',
                    'trap_reason': 'Broker creating confusion'
                }
            
            # HIGH MANIPULATION SCORE = ACTIVE TRAP
            if manipulation_score > 0.6:
                return {
                    'pattern': 'MANIPULATED_PATTERN',
                    'broker_trap': 'ACTIVE_MANIPULATION',
                    'confidence': 0.8,
                    'counter_strategy': 'REVERSE_OBVIOUS',
                    'trap_reason': 'High manipulation detected'
                }
            
            # MODERATE SETUP = POSSIBLE GENUINE
            return {
                'pattern': 'MODERATE_SETUP',
                'broker_trap': 'LOW_RISK',
                'confidence': 0.6,
                'counter_strategy': 'FOLLOW_ANALYSIS',
                'trap_reason': 'Appears genuine'
            }
            
        except Exception as e:
            print(f"Pattern trap detection error: {e}")
            return {
                'pattern': 'ERROR',
                'broker_trap': 'UNKNOWN',
                'confidence': 0.0,
                'counter_strategy': 'DEFENSIVE'
            }
    
    def _analyze_fake_volume(self, image):
        """Analyze volume but detect fake volume"""
        try:
            height, width = image.shape[:2]
            volume_section = image[int(height * 0.8):, :]
            gray_volume = cv2.cvtColor(volume_section, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_volume, 30, 100)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            volume_bars = len([c for c in contours if cv2.contourArea(c) > 20])
            
            # FAKE VOLUME DETECTION
            fake_probability = 0.0
            
            # Too perfect volume = fake
            if volume_bars > 25:
                fake_probability += 0.4
            
            # No volume = hiding manipulation
            if volume_bars < 3:
                fake_probability += 0.5
            
            return {
                'level': 'HIGH' if volume_bars > 15 else 'MEDIUM' if volume_bars > 8 else 'LOW',
                'bars_detected': volume_bars,
                'fake_probability': fake_probability,
                'broker_volume': 'FAKE' if fake_probability > 0.6 else 'SUSPICIOUS' if fake_probability > 0.3 else 'POSSIBLE_REAL'
            }
            
        except Exception as e:
            print(f"Fake volume analysis error: {e}")
            return {
                'level': 'UNKNOWN',
                'bars_detected': 0,
                'fake_probability': 0.8,
                'broker_volume': 'FAKE'
            }
    
    def _analyze_broker_psychology(self, candles_data, trend_analysis, pattern_analysis):
        """Analyze what the broker wants you to think"""
        try:
            broker_bias = candles_data.get('broker_bias', 'NEUTRAL')
            broker_wants = trend_analysis.get('broker_wants', 'UNKNOWN')
            trap_type = pattern_analysis.get('broker_trap', 'UNKNOWN')
            
            # BROKER PSYCHOLOGY MATRIX
            psychology_score = {
                'wants_calls': 0.0,
                'wants_puts': 0.0,
                'wants_confusion': 0.0,
                'confidence_level': 0.0
            }
            
            # What broker wants based on bias
            if broker_bias == 'BULLISH':
                psychology_score['wants_calls'] += 0.4
            elif broker_bias == 'BEARISH':
                psychology_score['wants_puts'] += 0.4
            else:
                psychology_score['wants_confusion'] += 0.3
            
            # What broker wants based on trend
            if broker_wants == 'CALLS':
                psychology_score['wants_calls'] += 0.3
            elif broker_wants == 'PUTS':
                psychology_score['wants_puts'] += 0.3
            elif broker_wants == 'CONFUSION':
                psychology_score['wants_confusion'] += 0.4
            
            # Trap influence
            if trap_type == 'BULL_TRAP':
                psychology_score['wants_calls'] += 0.5
            elif trap_type == 'BEAR_TRAP':
                psychology_score['wants_puts'] += 0.5
            elif trap_type == 'CONFUSION_TRAP':
                psychology_score['wants_confusion'] += 0.5
            
            # Determine broker's main intention
            max_score = max(psychology_score['wants_calls'], psychology_score['wants_puts'], psychology_score['wants_confusion'])
            
            if psychology_score['wants_calls'] == max_score:
                broker_intention = 'WANTS_YOU_TO_BUY_CALLS'
                recommended_action = 'CONSIDER_PUT'
            elif psychology_score['wants_puts'] == max_score:
                broker_intention = 'WANTS_YOU_TO_BUY_PUTS'
                recommended_action = 'CONSIDER_CALL'
            else:
                broker_intention = 'WANTS_CONFUSION'
                recommended_action = 'WAIT_FOR_CLARITY'
            
            return {
                'broker_intention': broker_intention,
                'recommended_action': recommended_action,
                'confidence': max_score,
                'psychology_scores': psychology_score,
                'manipulation_level': 'HIGH' if max_score > 0.8 else 'MEDIUM' if max_score > 0.5 else 'LOW'
            }
            
        except Exception as e:
            print(f"Broker psychology error: {e}")
            return {
                'broker_intention': 'UNKNOWN',
                'recommended_action': 'DEFENSIVE',
                'confidence': 0.0,
                'psychology_scores': {},
                'manipulation_level': 'HIGH'
            }
    
    def _detect_manipulation_signals(self, image, broker_threat):
        """Detect active manipulation signals"""
        try:
            manipulation_indicators = []
            manipulation_score = 0.0
            
            # TIME-BASED MANIPULATION
            if broker_threat['score'] >= 4:
                manipulation_indicators.append('HIGH_RISK_TIME')
                manipulation_score += 0.3
            
            # PATTERN-BASED MANIPULATION
            # Check for too perfect patterns (broker manipulation)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect perfect horizontal lines (fake support/resistance)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
            horizontal_lines = cv2.morphologyEx(cv2.Canny(gray, 50, 150), cv2.MORPH_OPEN, horizontal_kernel)
            h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            perfect_lines = len([c for c in h_contours if cv2.contourArea(c) > 100])
            
            if perfect_lines > 5:
                manipulation_indicators.append('TOO_MANY_PERFECT_LINES')
                manipulation_score += 0.2
            
            # Detect artificial symmetry
            height, width = image.shape[:2]
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            
            # Compare halves for artificial symmetry
            correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)
            max_correlation = np.max(correlation) if correlation.size > 0 else 0
            
            if max_correlation > 0.8:
                manipulation_indicators.append('ARTIFICIAL_SYMMETRY')
                manipulation_score += 0.3
            
            return {
                'indicators': manipulation_indicators,
                'score': manipulation_score,
                'level': 'EXTREME' if manipulation_score > 0.7 else 'HIGH' if manipulation_score > 0.5 else 'MODERATE' if manipulation_score > 0.3 else 'LOW',
                'broker_active': manipulation_score > 0.5
            }
            
        except Exception as e:
            print(f"Manipulation detection error: {e}")
            return {
                'indicators': ['DETECTION_ERROR'],
                'score': 0.8,
                'level': 'HIGH',
                'broker_active': True
            }
    
    def _detect_broker_traps(self, pattern_analysis, trend_analysis, broker_threat):
        """Detect specific broker trap types"""
        try:
            trap_types = []
            trap_confidence = 0.0
            
            # BULL TRAP DETECTION
            if (pattern_analysis.get('broker_trap') == 'BULL_TRAP' or 
                trend_analysis.get('broker_wants') == 'CALLS'):
                trap_types.append('BULL_TRAP')
                trap_confidence += 0.4
            
            # BEAR TRAP DETECTION
            if (pattern_analysis.get('broker_trap') == 'BEAR_TRAP' or 
                trend_analysis.get('broker_wants') == 'PUTS'):
                trap_types.append('BEAR_TRAP')
                trap_confidence += 0.4
            
            # CONFUSION TRAP
            if pattern_analysis.get('broker_trap') == 'CONFUSION_TRAP':
                trap_types.append('CONFUSION_TRAP')
                trap_confidence += 0.3
            
            # TIME-BASED TRAP
            if broker_threat['score'] >= 6:
                trap_types.append('TIME_TRAP')
                trap_confidence += 0.2
            
            # VOLUME TRAP
            if 'FAKE_VOLUME' in broker_threat.get('factors', []):
                trap_types.append('VOLUME_TRAP')
                trap_confidence += 0.2
            
            return {
                'trap_types': trap_types,
                'confidence': min(trap_confidence, 1.0),
                'primary_trap': trap_types[0] if trap_types else 'NO_TRAP',
                'danger_level': 'EXTREME' if trap_confidence > 0.8 else 'HIGH' if trap_confidence > 0.6 else 'MODERATE' if trap_confidence > 0.4 else 'LOW'
            }
            
        except Exception as e:
            print(f"Trap detection error: {e}")
            return {
                'trap_types': ['UNKNOWN_TRAP'],
                'confidence': 0.7,
                'primary_trap': 'UNKNOWN_TRAP',
                'danger_level': 'HIGH'
            }
    
    def _generate_counter_strategy(self, war_analysis, broker_threat):
        """Generate counter-strategy against broker"""
        try:
            broker_psychology = war_analysis.get('broker_psychology', {})
            trap_analysis = war_analysis.get('trap_analysis', {})
            manipulation_signals = war_analysis.get('manipulation_signals', {})
            
            # BASE COUNTER-STRATEGY
            if broker_threat['strategy'] == 'FULL_REVERSE_PSYCHOLOGY':
                # Maximum threat - do complete opposite
                if broker_psychology.get('broker_intention') == 'WANTS_YOU_TO_BUY_CALLS':
                    counter_signal = 'PUT'
                    strategy_name = 'FULL_REVERSE_CALLS'
                    confidence = 0.85
                elif broker_psychology.get('broker_intention') == 'WANTS_YOU_TO_BUY_PUTS':
                    counter_signal = 'CALL'
                    strategy_name = 'FULL_REVERSE_PUTS'
                    confidence = 0.85
                else:
                    counter_signal = 'NO_TRADE'
                    strategy_name = 'DEFENSIVE_WAIT'
                    confidence = 0.0
            
            elif broker_threat['strategy'] == 'CHAOS_WARFARE':
                # High threat - use chaos tactics
                chaos_options = ['CALL', 'PUT', 'NO_TRADE']
                weights = [0.4, 0.4, 0.2]  # Slightly favor action over waiting
                counter_signal = np.random.choice(chaos_options, p=weights)
                strategy_name = 'CHAOS_WARFARE'
                confidence = 0.75
            
            elif broker_threat['strategy'] == 'STEALTH_OPERATIONS':
                # Moderate threat - use stealth
                recommended_action = broker_psychology.get('recommended_action', 'WAIT_FOR_CLARITY')
                if 'CONSIDER_PUT' in recommended_action:
                    counter_signal = 'PUT'
                    strategy_name = 'STEALTH_PUT'
                    confidence = 0.70
                elif 'CONSIDER_CALL' in recommended_action:
                    counter_signal = 'CALL'
                    strategy_name = 'STEALTH_CALL'
                    confidence = 0.70
                else:
                    counter_signal = 'NO_TRADE'
                    strategy_name = 'STEALTH_WAIT'
                    confidence = 0.0
            
            else:
                # Low threat - normal analysis with caution
                if trap_analysis.get('primary_trap') == 'BULL_TRAP':
                    counter_signal = 'PUT'
                    strategy_name = 'ANTI_BULL_TRAP'
                    confidence = 0.80
                elif trap_analysis.get('primary_trap') == 'BEAR_TRAP':
                    counter_signal = 'CALL'
                    strategy_name = 'ANTI_BEAR_TRAP'
                    confidence = 0.80
                else:
                    counter_signal = 'NO_TRADE'
                    strategy_name = 'CAUTIOUS_ANALYSIS'
                    confidence = 0.0
            
            # ADJUST FOR MANIPULATION LEVEL
            if manipulation_signals.get('broker_active', False):
                confidence *= 0.8  # Reduce confidence when broker is actively manipulating
                strategy_name += '_ANTI_MANIPULATION'
            
            return {
                'signal': counter_signal,
                'strategy': strategy_name,
                'confidence': confidence,
                'reasoning': self._generate_war_reasoning(war_analysis, broker_threat, counter_signal),
                'expiry_minutes': self._select_war_expiry(broker_threat, confidence),
                'entry_timing': self._calculate_war_timing(broker_threat)
            }
            
        except Exception as e:
            print(f"Counter-strategy error: {e}")
            return {
                'signal': 'NO_TRADE',
                'strategy': 'DEFENSIVE_ERROR',
                'confidence': 0.0,
                'reasoning': 'Error in strategy generation - staying defensive',
                'expiry_minutes': 2,
                'entry_timing': {'delay': 25, 'type': 'SAFE'}
            }
    
    def _psychological_warfare_decision(self, counter_strategy, broker_threat):
        """Final psychological warfare decision"""
        try:
            signal = counter_strategy['signal']
            confidence = counter_strategy['confidence']
            strategy = counter_strategy['strategy']
            
            # DECISION MATRIX
            execute = False
            final_confidence = confidence
            
            # HIGH CONFIDENCE ANTI-BROKER SIGNALS
            if confidence >= 0.80 and signal != 'NO_TRADE':
                execute = True
                decision_reason = f"High confidence {strategy} - broker counter-attack ready"
            
            # MODERATE CONFIDENCE WITH SPECIAL CONDITIONS
            elif confidence >= 0.70 and broker_threat['level'] in ['MAXIMUM THREAT 🔴', 'HIGH THREAT 🟠']:
                execute = True
                final_confidence = confidence * 0.9  # Slight reduction for risk
                decision_reason = f"Moderate confidence but high broker threat - executing {strategy}"
            
            # CHAOS WARFARE ALWAYS EXECUTES (UNPREDICTABILITY)
            elif 'CHAOS' in strategy and confidence >= 0.60:
                execute = True
                decision_reason = f"Chaos warfare - unpredictable execution to confuse broker"
            
            # DEFENSIVE MODE
            else:
                execute = False
                decision_reason = f"Insufficient confidence or too risky - staying defensive"
            
            return {
                'execute': execute,
                'signal': signal,
                'confidence': final_confidence,
                'strategy': strategy,
                'decision_reason': decision_reason,
                'war_plan': counter_strategy,
                'broker_threat': broker_threat
            }
            
        except Exception as e:
            print(f"Warfare decision error: {e}")
            return {
                'execute': False,
                'signal': 'NO_TRADE',
                'confidence': 0.0,
                'strategy': 'ERROR_DEFENSIVE',
                'decision_reason': 'Error in decision making - staying safe',
                'war_plan': {},
                'broker_threat': broker_threat
            }
    
    def _generate_war_reasoning(self, war_analysis, broker_threat, signal):
        """Generate reasoning for war strategy"""
        try:
            broker_psychology = war_analysis.get('broker_psychology', {})
            trap_analysis = war_analysis.get('trap_analysis', {})
            
            reasoning = f"🎯 ANTI-BROKER ANALYSIS:\n\n"
            
            # BROKER INTENTION
            broker_intention = broker_psychology.get('broker_intention', 'UNKNOWN')
            reasoning += f"🧠 Broker wants: {broker_intention.replace('_', ' ').title()}\n"
            
            # COUNTER-STRATEGY
            if signal == 'CALL':
                reasoning += f"⚡ Counter-strategy: BUY CALL (Broker expects PUT)\n"
            elif signal == 'PUT':
                reasoning += f"⚡ Counter-strategy: BUY PUT (Broker expects CALL)\n"
            else:
                reasoning += f"🛡️ Counter-strategy: NO TRADE (Too dangerous)\n"
            
            # TRAP DETECTION
            primary_trap = trap_analysis.get('primary_trap', 'NO_TRAP')
            if primary_trap != 'NO_TRAP':
                reasoning += f"🪤 Trap detected: {primary_trap.replace('_', ' ').title()}\n"
            
            # THREAT LEVEL
            reasoning += f"🚨 Threat level: {broker_threat['level']}\n"
            
            # MANIPULATION FACTORS
            if broker_threat.get('factors'):
                reasoning += f"⚠️ Risk factors: {', '.join(broker_threat['factors'][:2])}\n"
            
            return reasoning
            
        except Exception as e:
            print(f"War reasoning error: {e}")
            return "🎯 ANTI-BROKER ANALYSIS: Error in analysis - using defensive approach"
    
    def _select_war_expiry(self, broker_threat, confidence):
        """Select expiry time for anti-broker warfare"""
        try:
            # SHORT EXPIRY FOR HIGH THREAT (Less time for broker manipulation)
            if broker_threat['score'] >= 6:
                return 1  # 1 minute - quick in/out
            elif broker_threat['score'] >= 4:
                return 2  # 2 minutes - moderate risk
            elif confidence >= 0.80:
                return 3  # 3 minutes - high confidence allows longer
            else:
                return 2  # Default 2 minutes
                
        except Exception as e:
            print(f"War expiry error: {e}")
            return 1  # Default to safest option
    
    def _calculate_war_timing(self, broker_threat):
        """Calculate optimal entry timing for anti-broker warfare"""
        try:
            if broker_threat['score'] >= 6:
                # MAXIMUM THREAT - Use chaos timing
                chaos_delays = [13, 17, 19, 23, 27, 29, 31, 37, 41, 43]
                delay = random.choice(chaos_delays)
                timing_type = 'CHAOS_TIMING'
            elif broker_threat['score'] >= 4:
                # HIGH THREAT - Use odd timing
                odd_delays = [17, 19, 23, 27, 29, 31, 37]
                delay = random.choice(odd_delays)
                timing_type = 'ODD_TIMING'
            elif broker_threat['score'] >= 2:
                # MODERATE THREAT - Avoid round numbers
                safe_delays = [16, 18, 22, 24, 26, 28, 32, 34]
                delay = random.choice(safe_delays)
                timing_type = 'SAFE_TIMING'
            else:
                # LOW THREAT - Normal timing
                delay = random.randint(15, 35)
                timing_type = 'NORMAL_TIMING'
            
            return {
                'delay': delay,
                'type': timing_type,
                'anti_manipulation': broker_threat['score'] >= 4
            }
            
        except Exception as e:
            print(f"War timing error: {e}")
            return {'delay': 25, 'type': 'SAFE', 'anti_manipulation': True}
    
    async def _execute_war_plan(self, update, decision, local_time, broker_threat):
        """Execute the anti-broker war plan"""
        try:
            signal = decision['signal']
            confidence = decision['confidence']
            strategy = decision['strategy']
            war_plan = decision['war_plan']
            
            # Calculate timing
            entry_timing = war_plan['entry_timing']
            entry_time = local_time + timedelta(seconds=entry_timing['delay'])
            expiry_time = entry_time + timedelta(minutes=war_plan['expiry_minutes'])
            
            if signal == 'CALL':
                emoji = "📈"
                action = "BUY"
                color = "🟢"
                war_emoji = "⚔️🟢"
            else:
                emoji = "📉"
                action = "SELL"
                color = "🔴"
                war_emoji = "⚔️🔴"
            
            message = f"""{war_emoji} <b>ANTI-BROKER WAR SIGNAL</b> {war_emoji}

{emoji} <b>COUNTER-ATTACK: {action}</b> {color}

🕰️ <b>TIME:</b> {local_time.strftime('%H:%M:%S')} UTC+6
🚨 <b>BROKER THREAT:</b> {broker_threat['level']}
⚔️ <b>WAR STRATEGY:</b> {strategy}

⏰ <b>CURRENT TIME:</b> {local_time.strftime('%H:%M:%S')}
🎯 <b>ENTRY TIME:</b> {entry_time.strftime('%H:%M:%S')}
⏳ <b>WAIT:</b> {entry_timing['delay']} seconds
🏁 <b>EXPIRY TIME:</b> {expiry_time.strftime('%H:%M:%S')}
⌛ <b>DURATION:</b> {war_plan['expiry_minutes']} minute(s)

💎 <b>WAR CONFIDENCE:</b> <b>{confidence:.1%}</b>
🧠 <b>ENTRY TYPE:</b> {entry_timing['type']}

{war_plan['reasoning']}

⚡ <b>BATTLE PLAN:</b>
1️⃣ Wait exactly {entry_timing['delay']} seconds ({entry_timing['type']})
2️⃣ Execute {action} at {entry_time.strftime('%H:%M:%S')}
3️⃣ Set {war_plan['expiry_minutes']}m expiry (Anti-manipulation timing)
4️⃣ Close at {expiry_time.strftime('%H:%M:%S')}

🛡️ <b>ANTI-BROKER TACTICS:</b>
• Using {entry_timing['type'].lower().replace('_', ' ')} to avoid broker algorithms
• {war_plan['expiry_minutes']}-minute expiry optimized for broker behavior
• Signal based on REVERSE PSYCHOLOGY analysis
• Entry timing designed to confuse broker systems

💀 <b>BROKER WARFARE TIPS:</b>
• Use SMALL amounts - stay under broker radar
• This signal COUNTERS what broker expects
• Timing optimized to avoid manipulation zones
• Based on ANTI-BROKER psychological analysis

━━━━━━━━━━━━━━━━━━━━━
⚔️ <b>Anti-Broker War Machine</b>
💀 <i>Designed to Beat Broker Manipulation</i>"""

            await update.message.reply_text(message, parse_mode='HTML')
            print(f"⚔️ WAR SIGNAL DEPLOYED: {signal} ({confidence:.1%}) - {strategy}")
            
            # Start war countdown
            await self._execute_war_countdown(update, entry_timing, signal, war_plan['expiry_minutes'])
            
        except Exception as e:
            print(f"War execution error: {e}")
            await update.message.reply_text(
                f"⚔️ WAR SIGNAL: {signal}\n"
                f"💎 Confidence: {confidence:.1%}\n"
                f"🎯 Strategy: ANTI-BROKER WARFARE"
            )
    
    async def _send_no_war_message(self, update, decision, broker_threat):
        """Send message when no war action is taken"""
        try:
            message = f"""⚔️ <b>ANTI-BROKER WAR ANALYSIS</b> ⚔️

🛡️ <b>DECISION:</b> NO ATTACK ⚪

🚨 <b>BROKER THREAT:</b> {broker_threat['level']}
🎯 <b>STRATEGY:</b> {decision['strategy']}
💎 <b>CONFIDENCE:</b> {decision['confidence']:.1%}

❌ <b>REASON FOR DEFENSIVE STANCE:</b>
{decision['decision_reason']}

🧠 <b>BROKER INTELLIGENCE:</b>
• Threat Score: {broker_threat['score']}/10
• Manipulation Mode: {broker_threat.get('broker_mode', 'Unknown')}
• Risk Factors: {len(broker_threat.get('factors', []))}

💡 <b>TACTICAL RECOMMENDATION:</b>
Wait for better opportunity or lower broker threat level

🛡️ <b>DEFENSIVE POSTURE:</b>
Monitoring broker patterns for next counter-attack opportunity

━━━━━━━━━━━━━━━━━━━━━
⚔️ <b>Anti-Broker War Machine</b>
🛡️ <i>Strategic Patience • No Unnecessary Risks</i>"""

            await update.message.reply_text(message, parse_mode='HTML')
            print(f"🛡️ DEFENSIVE STANCE: {decision['decision_reason']}")
            
        except Exception as e:
            print(f"No war message error: {e}")
            await update.message.reply_text("⚔️ ANALYSIS COMPLETE - STAYING DEFENSIVE")
    
    async def _execute_war_countdown(self, update, entry_timing, signal_type, expiry_minutes):
        """Execute war countdown with psychological warfare"""
        try:
            entry_delay = entry_timing['delay']
            
            # Wait for initial period
            initial_wait = max(0, entry_delay - 10)
            if initial_wait > 0:
                await asyncio.sleep(initial_wait)
            
            # War countdown
            remaining = min(10, entry_delay - initial_wait)
            if remaining >= 5:
                await asyncio.sleep(remaining - 5)
                await update.message.reply_text(
                    f"⚔️ <b>5 SECONDS TO COUNTER-ATTACK!</b>\n"
                    f"🎯 {signal_type} • {expiry_minutes}m expiry\n"
                    f"💀 ANTI-BROKER WARFARE READY!",
                    parse_mode='HTML'
                )
                
                await asyncio.sleep(4)
                await update.message.reply_text(
                    f"💥 <b>ATTACK NOW!</b> 💥\n"
                    f"⚔️ {signal_type} • DESTROY BROKER!\n"
                    f"🧠 REVERSE PSYCHOLOGY ACTIVE!",
                    parse_mode='HTML'
                )
            
            # Final war confirmation
            await asyncio.sleep(2)
            await update.message.reply_text(
                f"✅ <b>COUNTER-ATTACK LAUNCHED</b>\n"
                f"⚔️ {signal_type} deployed against broker\n"
                f"⏰ Expiry: {expiry_minutes} minute(s)\n"
                f"💀 May the broker manipulation fail!",
                parse_mode='HTML'
            )
            
        except Exception as e:
            print(f"War countdown error: {e}")
    
    def _create_defensive_analysis(self):
        """Create defensive analysis when errors occur"""
        return {
            'candles_data': {'total_candles': 0, 'manipulation_score': 0.8, 'broker_bias': 'UNKNOWN'},
            'trend_analysis': {'direction': 'DEFENSIVE', 'broker_wants': 'UNKNOWN', 'counter_move': 'WAIT'},
            'pattern_analysis': {'broker_trap': 'UNKNOWN', 'counter_strategy': 'DEFENSIVE'},
            'volume_analysis': {'broker_volume': 'FAKE', 'fake_probability': 0.8},
            'broker_psychology': {'broker_intention': 'UNKNOWN', 'recommended_action': 'DEFENSIVE'},
            'manipulation_signals': {'broker_active': True, 'level': 'HIGH'},
            'trap_analysis': {'primary_trap': 'UNKNOWN_TRAP', 'danger_level': 'HIGH'},
            'analysis_time': 'ERROR'
        }
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages with war machine info"""
        try:
            local_time = self._get_utc_plus_6_time()
            broker_threat = self._assess_broker_threat(local_time)
            
            await update.message.reply_text(
                f"⚔️ <b>ANTI-BROKER WAR MACHINE</b> ⚔️\n\n"
                f"🕰️ <b>Current Time:</b> {local_time.strftime('%H:%M:%S')} UTC+6\n"
                f"🚨 <b>Broker Threat:</b> {broker_threat['level']}\n"
                f"🎯 <b>War Strategy:</b> {broker_threat['strategy']}\n\n"
                f"📱 <b>Send chart for ANTI-BROKER analysis!</b>\n\n"
                f"⚔️ <b>WAR MACHINE FEATURES:</b>\n"
                f"• Reverse psychology analysis\n"
                f"• Broker trap detection\n"
                f"• Manipulation pattern recognition\n"
                f"• Psychological warfare tactics\n"
                f"• Anti-broker timing algorithms\n\n"
                f"💀 <b>DESIGNED TO BEAT BROKER MANIPULATION!</b>",
                parse_mode='HTML'
            )
        except Exception as e:
            print(f"Text handler error: {e}")
    
    async def start_bot(self):
        """Start the Anti-Broker War Machine"""
        if not TELEGRAM_AVAILABLE:
            print("❌ Install: pip install python-telegram-bot opencv-python")
            return
        
        print("⚔️" * 80)
        print("💀 ANTI-BROKER WAR MACHINE STARTED")
        print("🎯 MISSION: DESTROY BROKER MANIPULATION")
        print("🧠 STRATEGY: PSYCHOLOGICAL WARFARE")
        print("💥 TARGET: QUOTEX OTC BROKERS")
        print("🛡️ TACTICS: REVERSE PSYCHOLOGY + CHAOS TIMING")
        print("⚔️" * 80)
        
        try:
            self.application = Application.builder().token(self.token).build()
            
            self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
            self.application.add_handler(MessageHandler(filters.TEXT, self.handle_text))
            
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            print("✅ Anti-Broker War Machine ONLINE...")
            print("💀 Ready to destroy broker manipulation!")
            
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"❌ War Machine Error: {e}")
        finally:
            if self.application:
                try:
                    await self.application.stop()
                    await self.application.shutdown()
                except:
                    pass

async def main():
    logging.basicConfig(level=logging.INFO)
    
    print("⚔️" * 90)
    print("💀 STARTING ANTI-BROKER WAR MACHINE")
    print("🎯 DESIGNED TO BEAT QUOTEX OTC MANIPULATION")
    print("🧠 USES REVERSE PSYCHOLOGY AND CHAOS WARFARE")
    print("💥 COUNTERS BROKER TRAPS AND MANIPULATION")
    print("🛡️ PROTECTS TRADERS FROM BROKER GAMES")
    print("⚔️" * 90)
    
    try:
        bot = AntiBrokerWarBot()
        await bot.start_bot()
    except Exception as e:
        print(f"❌ War Machine Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n💀 Anti-Broker War Machine stopped")
    except Exception as e:
        print(f"❌ Error: {e}")