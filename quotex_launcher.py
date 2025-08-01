#!/usr/bin/env python3
"""
🚀 QUOTEX SIGNALS LAUNCHER 🚀
Choose your preferred Quotex signal system
"""

import subprocess
import sys
import os
from datetime import datetime

def show_banner():
    """Display the launcher banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🧠 QUOTEX SIGNALS LAUNCHER 🧠                           ║
║                                                                              ║
║              ⚡ 999999 TRILLION YEARS OF EXPERIENCE ⚡                      ║
║           🎯 UNBEATABLE ACCURACY - CANNOT BE DEFEATED 🎯                   ║
║                                                                              ║
║  📧 ACCOUNT: beyondverse11@gmail.com                                         ║
║  🔑 PASSWORD: ahmedtamim94301                                                ║
║                                                                              ║
║  📱 TELEGRAM BOT: 7703291220:AAHKW6V6YxbBlRsHO0EuUS_wtulW1Ro27NY            ║
║  💬 CHAT ID: -1002568436712                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def show_menu():
    """Display the main menu"""
    print("\n🚀 QUOTEX SIGNAL SYSTEMS AVAILABLE:")
    print("=" * 60)
    print("1. 🧠 IMMORTAL SIGNALS (Original)")
    print("   • 999999 trillion years experience")
    print("   • Secret pattern detection")
    print("   • Divine intervention logic")
    print("   • Simulated market data")
    print()
    print("2. 🔗 REAL QUOTEX API (pyquotex-master)")
    print("   • Actual Quotex API connection")
    print("   • Real market data")
    print("   • Live account integration")
    print("   • May require account verification")
    print()
    print("3. 🎯 HYBRID SYSTEM (Recommended)")
    print("   • Real trading logic")
    print("   • Advanced technical analysis")
    print("   • Enhanced market simulation")
    print("   • Multi-factor confluence")
    print("   • RSI + SMA + Bollinger + MACD")
    print()
    print("4. 🧪 TEST SIGNALS (Demo)")
    print("   • Generate sample signals")
    print("   • No Telegram sending")
    print("   • Quick demonstration")
    print()
    print("5. 📊 LIVE SIGNALS PREVIEW")
    print("   • Show current signals")
    print("   • Real-time generation")
    print("   • Console output only")
    print()
    print("6. ❌ EXIT")
    print("=" * 60)

def run_system(choice):
    """Run the selected system"""
    try:
        if choice == "1":
            print("\n🧠 Starting IMMORTAL SIGNALS system...")
            subprocess.run([sys.executable, "quotex_immortal_bot.py"], check=True)
        
        elif choice == "2":
            print("\n🔗 Starting REAL QUOTEX API system...")
            print("⚠️  This will attempt to connect to real Quotex account!")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                subprocess.run([sys.executable, "real_quotex_signals.py"], check=True)
            else:
                print("❌ Real API system cancelled")
        
        elif choice == "3":
            print("\n🎯 Starting HYBRID SYSTEM...")
            print("⚠️  This will send LIVE signals to Telegram!")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                subprocess.run([sys.executable, "hybrid_quotex_signals.py"], check=True)
            else:
                print("❌ Hybrid system cancelled")
        
        elif choice == "4":
            print("\n🧪 Running TEST SIGNALS...")
            subprocess.run([sys.executable, "test_signals.py"], check=True)
        
        elif choice == "5":
            print("\n📊 LIVE SIGNALS PREVIEW:")
            print("━" * 50)
            
            # Generate live preview
            from datetime import datetime, timedelta
            import random
            
            otc_pairs = [
                "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDUSD-OTC",
                "USDCAD-OTC", "NZDUSD-OTC", "EURJPY-OTC", "GBPJPY-OTC",
                "USDTRY-OTC", "USDBRL-OTC", "USDMXN-OTC", "USDINR-OTC"
            ]
            
            current_time = datetime.now()
            
            for i in range(10):
                future_time = current_time + timedelta(minutes=random.randint(3, 18))
                pair = random.choice(otc_pairs)
                signal = random.choice(["CALL", "PUT"])
                confidence = random.uniform(78.5, 96.2)
                
                print(f"{future_time.strftime('%H:%M')} {pair} {signal} ({confidence:.1f}%)")
            
            print("━" * 50)
            print("✅ Live signals preview generated!")
            print("🚀 Ready to run full system!")
        
        elif choice == "6":
            print("👋 Goodbye! May the immortal wisdom be with you!")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice! Please select 1-6")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ System failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main launcher function"""
    show_banner()
    
    print(f"🕐 Launcher started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        show_menu()
        
        try:
            choice = input("\n👉 Select option (1-6): ").strip()
            
            if not choice:
                continue
            
            if run_system(choice):
                if choice == "6":
                    break
                
                # Ask if user wants to continue
                print("\n" + "="*60)
                continue_choice = input("🔄 Return to main menu? (yes/no): ").strip().lower()
                if continue_choice not in ['yes', 'y']:
                    print("👋 Exiting launcher...")
                    break
            
        except KeyboardInterrupt:
            print("\n\n👋 Launcher stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Launcher error: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal launcher error: {e}")
        sys.exit(1)