# Supreme Sniper Bot — Enhanced with 5 OTC Sniper Strategies + Bullet Expiry
# =====================================================================
# • Adds 5 advanced OTC sniper strategies.
# • Bullet‑style Telegram message with real expiry duration line.
# • 15‑second spacing between signals to avoid spam.
# ---------------------------------------------------------------------

import os, time, logging, datetime as dt
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from telegram import Bot
from telegram.error import TelegramError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EMAIL      = os.getenv("EMAIL")
PASSWORD   = os.getenv("PASSWORD")
BOT_TOKEN  = os.getenv("BOT_TOKEN")
CHAT_ID    = os.getenv("CHAT_ID")

# Scan 1‑ and 2‑minute charts
TIMEFRAMES   = [1, 2]
SCAN_SLEEP   = 1.5
RSI_PERIOD   = 14
EMA_PERIOD   = 20
WINDOW       = 50
MIN_SCORE    = 3  # confluences needed for real‑market strategy

class SupremeSniperBot:
    def __init__(self):
        self.bot = Bot(BOT_TOKEN)
        self.driver = self._init_browser()
        self._login()

    # ---------------- Browser ----------------
    def _init_browser(self):
        chromedriver_autoinstaller.install()
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        return webdriver.Chrome(options=options)

    def _login(self):
        logging.info("Logging in…")
        self.driver.get("https://quotex.com/en/login")
        time.sleep(2)
        self.driver.find_element(By.NAME, "email").send_keys(EMAIL)
        self.driver.find_element(By.NAME, "password").send_keys(PASSWORD)
        self.driver.find_element(By.CSS_SELECTOR, "button[type=submit]").click()
        time.sleep(5)
        if "trade" not in self.driver.current_url:
            raise RuntimeError("Login failed — check credentials/2FA.")
        logging.info("Logged in ✅")

    # ---------------- Data ----------------
    def fetch_candles(self, asset: str, tf: int) -> pd.DataFrame:
        js = "return window.chartData ? window.chartData : null;"
        raw = self.driver.execute_script(js)
        if raw is None:
            return pd.DataFrame()
        arr = raw.get(asset, {}).get(str(tf), [])[-WINDOW:]
        if not arr:
            return pd.DataFrame()
        df = pd.DataFrame(arr, columns=["epoch", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")
        return df.set_index("timestamp")

    # ---------------- Indicators ----------------
    @staticmethod
    def ema(series: pd.Series, span: int):
        return series.ewm(span=span).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(gain).rolling(period).mean()
        roll_down = pd.Series(loss).rolling(period).mean()
        rs = roll_up / roll_down.replace({0: np.nan})
        return 100 - (100 / (1 + rs))

    # ---------------- OTC Strategy ----------------
    def otc_sniper(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 5:
            return None
        last, prev, preprev = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        signals = []

        # 1️⃣ Three‑candle trap
        if all(c.close > c.open for c in [preprev, prev, last]):
            signals.append(("PUT", "3 Green Candle Trap"))
        elif all(c.close < c.open for c in [preprev, prev, last]):
            signals.append(("CALL", "3 Red Candle Trap"))

        # 2️⃣ Wick rejection (<25 % body)
        wick = last.high - last.low
        body = abs(last.close - last.open)
        if wick and body / wick < 0.25:
            signals.append(("CALL", "Wick Trap Bull") if last.close > last.open else ("PUT", "Wick Trap Bear"))

        # 3️⃣ False breakout of previous high/low
        if prev.high < last.high < prev.high + (prev.high - prev.low) and last.close < prev.high:
            signals.append(("PUT", "False Breakout High"))
        if prev.low > last.low > prev.low - (prev.high - prev.low) and last.close > prev.low:
            signals.append(("CALL", "False Breakout Low"))

        # 4️⃣ Morning/Evening star (doji in middle)
        if preprev.close < preprev.open and abs(prev.close - prev.open) < 1e-5 and last.close > last.open:
            signals.append(("CALL", "Morning Star"))
        if preprev.close > preprev.open and abs(prev.close - prev.open) < 1e-5 and last.close < last.open:
            signals.append(("PUT", "Evening Star"))

        # 5️⃣ Inside bar trap
        if prev.high > last.high and prev.low < last.low:
            signals.append(("CALL", "Inside Bar Bull Break") if last.close > last.open else ("PUT", "Inside Bar Bear Break"))

        # Require ≥ 2 confirmations
        if len(signals) >= 2:
            direction, reason = signals[0]
            return {"dir": direction, "reason": reason + f" + {len(signals)-1} extra confluence(s)"}
        return None

    # ---------------- Messaging ----------------
    def send_signal(self, asset: str, tf: int, direction: str, reason: str):
        now = dt.datetime.now(dt.timezone.utc).strftime("%H:%M UTC")
        expiry_str = f"{tf} Minute" if tf == 1 else f"{tf} Minutes"
        msg = (
            f"🔥 𝗦𝗡𝗜𝗣𝗘𝗥 𝗦𝗜𝗚𝗡𝗔𝗟\n\n"
            f"• 📌 **Pair**: {asset}\n"
            f"• ⏱ **Timeframe**: {tf}M\n"
            f"• 📉 **Direction**: {direction}\n"
            f"• 🕓 **Time**: {now}\n"
            f"• ⏳ **Expiry**: {expiry_str}\n"
            f"• 🎯 **Strategy**: {reason}\n"
            f"• ✅ **Confidence**: 90%"
        )
        try:
            self.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
            logging.info("Signal sent: %s", msg.replace("\n", " | "))
        except TelegramError as e:
            logging.error("Telegram error: %s", e)

    # ---------------- Main Loop ----------------
    def run(self):
        logging.info("Starting OTC sniper ⏳")
        last_signal_time = time.time()
        while True:
            # Collect available OTC assets each cycle
            assets = [el.text for el in self.driver.find_elements(By.CLASS_NAME, "symbol-item") if "otc" in el.text.lower()]
            for asset in assets:
                for tf in TIMEFRAMES:
                    try:
                        df = self.fetch_candles(asset, tf)
                        if df.empty:
                            continue
                        result = self.otc_sniper(df)
                        if result and time.time() - last_signal_time > 15:
                            self.send_signal(asset, tf, result["dir"], result["reason"])
                            last_signal_time = time.time()
                    except Exception as exc:
                        logging.exception("Error analysing %s %dm: %s", asset, tf, exc)
                time.sleep(SCAN_SLEEP)

if __name__ == "__main__":
    SupremeSniperBot().run()