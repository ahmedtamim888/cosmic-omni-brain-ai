#!/usr/bin/env python3
"""
AI Supreme Sniper Monitor
=========================
Monitor the live AI engine status and performance.
"""

import os
import time
import subprocess
import psutil
from datetime import datetime

def load_env():
    """Load environment variables"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except:
        pass

def check_ai_process():
    """Check if AI Supreme Sniper is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'])
            if 'supreme_ai_sniper' in cmdline or 'run_ai_supreme_sniper' in cmdline:
                return proc.info
        except:
            continue
    return None

def show_system_status():
    """Show current system status"""
    print("🤖 AI Supreme Sniper - Live Market Monitor")
    print("=" * 60)
    print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Check AI process
    ai_proc = check_ai_process()
    if ai_proc:
        print(f"✅ AI Engine Status: RUNNING (PID: {ai_proc['pid']})")
        
        # Get process details
        try:
            proc = psutil.Process(ai_proc['pid'])
            cpu_percent = proc.cpu_percent()
            memory_mb = proc.memory_info().rss / 1024 / 1024
            print(f"📊 CPU Usage: {cpu_percent:.1f}%")
            print(f"🧠 Memory Usage: {memory_mb:.1f} MB")
            print(f"⏱ Runtime: {time.time() - proc.create_time():.0f} seconds")
        except:
            pass
    else:
        print("❌ AI Engine Status: NOT RUNNING")
    
    # Check environment
    load_env()
    email = os.getenv('EMAIL', 'Not Set')
    bot_token = os.getenv('BOT_TOKEN', 'Not Set')
    chat_id = os.getenv('CHAT_ID', 'Not Set')
    
    print(f"\n🔧 Configuration:")
    print(f"   📧 Email: {email[:10]}...")
    print(f"   🤖 Bot Token: {bot_token[:10]}...")
    print(f"   💬 Chat ID: {chat_id}")
    
    print(f"\n📈 Market Status:")
    print(f"   🌐 Platform: Quotex Live")
    print(f"   📊 Markets: OTC Currency Pairs")
    print(f"   ⏰ Timeframes: 1M, 2M")
    print(f"   🎯 Confidence: 85% Minimum")

def main():
    """Main monitoring function"""
    print("🚀 Starting AI Supreme Sniper Monitor...")
    
    while True:
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            show_system_status()
            
            ai_proc = check_ai_process()
            if not ai_proc:
                print(f"\n⚠️  AI Engine not running!")
                print(f"💡 To start: python3 run_ai_supreme_sniper.py")
                
                response = input(f"\n🔄 Start AI Engine now? (y/n): ").lower()
                if response == 'y':
                    print("🚀 Launching AI Supreme Sniper...")
                    subprocess.Popen(['python3', 'run_ai_supreme_sniper.py'])
                    time.sleep(5)
                    continue
            
            print(f"\n" + "=" * 60)
            print(f"🔄 Refreshing in 30 seconds... (Ctrl+C to exit)")
            print(f"📱 Check your Telegram for live signals!")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitor stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()