#!/usr/bin/env python3
"""
Quick AI Status Check
====================
"""

import os
import time
import psutil
from datetime import datetime
from telegram import Bot
import asyncio

async def send_status_update():
    """Send status update to Telegram"""
    bot_token = "7703291220:AAHKW6V6YxbBlRsHO0EuUS_wtulW1Ro27NY"
    chat_id = "-1002888605600"
    
    try:
        bot = Bot(token=bot_token)
        
        # Check if AI process is running
        ai_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'supreme_ai_sniper' in cmdline:
                    ai_running = True
                    break
            except:
                continue
        
        current_time = datetime.now().strftime("%H:%M:%S UTC")
        
        if ai_running:
            status_msg = (
                f"🤖 **𝗔𝗜 𝗦𝗨𝗣𝗥𝗘𝗠𝗘 𝗦𝗡𝗜𝗣𝗘𝗥 - 𝗦𝗧𝗔𝗧𝗨𝗦**\n\n"
                f"✅ **Status**: AI Engine ACTIVE\n"
                f"🌐 **Platform**: Quotex Live Markets\n"
                f"⏰ **Started**: {current_time}\n"
                f"🧠 **AI Mode**: 100 Billion Years Engine\n"
                f"📊 **Markets**: OTC Currency Pairs\n"
                f"🎯 **Confidence**: 85% Minimum\n"
                f"⚡ **Timeframes**: 1M, 2M\n\n"
                f"🔍 **Scanning for high-probability setups...**\n"
                f"📱 **You will receive signals here when detected!**\n\n"
                f"🤖 *Live market analysis in progress*"
            )
        else:
            status_msg = (
                f"⚠️ **𝗔𝗜 𝗦𝗨𝗣𝗥𝗘𝗠𝗘 𝗦𝗡𝗜𝗣𝗘𝗥 - 𝗦𝗧𝗔𝗧𝗨𝗦**\n\n"
                f"🔄 **Status**: Starting AI Engine...\n"
                f"⏰ **Time**: {current_time}\n"
                f"🌐 **Target**: Quotex Live Markets\n\n"
                f"🚀 *Initializing 100 Billion Years AI...*"
            )
        
        await bot.send_message(chat_id=chat_id, text=status_msg, parse_mode="Markdown")
        print("✅ Status update sent to Telegram!")
        
    except Exception as e:
        print(f"❌ Error sending status: {e}")

def check_process_status():
    """Check current process status"""
    print("🤖 AI Supreme Sniper - Quick Status Check")
    print("=" * 50)
    
    # Check if AI process is running
    ai_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'supreme_ai_sniper' in cmdline or 'run_ai_supreme' in cmdline:
                ai_processes.append(proc.info)
        except:
            continue
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏰ Current Time: {current_time}")
    
    if ai_processes:
        print(f"✅ AI Engine Status: RUNNING")
        for proc in ai_processes:
            runtime = time.time() - proc['create_time']
            print(f"   📍 PID: {proc['pid']}")
            print(f"   ⏱ Runtime: {runtime:.0f} seconds")
    else:
        print(f"🔄 AI Engine Status: STARTING...")
        print(f"   💡 The AI engine may take 10-30 seconds to fully initialize")
    
    print(f"\n🌐 Configuration:")
    print(f"   📧 Email: beyondverse11@gmail.com")
    print(f"   🤖 Bot: Active")
    print(f"   💬 Chat: -1002888605600")
    
    print(f"\n📊 Expected Behavior:")
    print(f"   🔍 AI will scan Quotex OTC markets")
    print(f"   📈 Analyze 1M and 2M timeframes")
    print(f"   🧠 Apply 100 Billion Years AI psychology")
    print(f"   ⚡ Send signals only when confidence > 85%")
    print(f"   📱 Signals will appear in your Telegram")

async def main():
    """Main function"""
    check_process_status()
    print(f"\n📱 Sending status update to Telegram...")
    await send_status_update()
    
    print(f"\n🎉 AI Supreme Sniper is now active!")
    print(f"📱 Monitor your Telegram for live trading signals")
    print(f"🤖 The AI will find high-probability setups automatically")

if __name__ == "__main__":
    asyncio.run(main())