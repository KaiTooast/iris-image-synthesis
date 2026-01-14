#!/usr/bin/env python3
"""
I.R.I.S. Universal Starter
==========================
Intelligent Rendering & Image Synthesis

Usage:
    python src/start.py                 # Default: --mode html
    python src/start.py --mode api      # API only (no frontend)
    python src/start.py --mode html     # API + HTML frontend
    python src/start.py --mode react    # API + React static build
    python src/start.py --mode full     # API + HTML + React
    python src/start.py --no-bot        # Disable Discord bot

Modes:
    api     - Only API server, no frontend (for external frontends)
    html    - API + classic HTML frontend (default)
    react   - API + React production build (served from frontend-react/dist)
    full    - API + HTML + React (both frontends available)

Press CTRL+C to exit | CTRL+R to restart (Windows)
"""

import sys
import subprocess
import os
import signal
import time
import json
import threading
import argparse
from pathlib import Path

# Windows keyboard input
if sys.platform == 'win32':
    import msvcrt

current_processes = []
shutdown_in_progress = False
restart_requested = False

def signal_handler(sig, frame):
    """Handle CTRL+C gracefully"""
    global shutdown_in_progress
    
    if shutdown_in_progress:
        print("\n[IRIS] Force shutdown...")
        for process in current_processes:
            try:
                process.kill()
            except:
                pass
        sys.exit(1)
    
    shutdown_in_progress = True
    print("\n\n[IRIS] Shutting down all services...")
    print("[IRIS] Press CTRL+C again to force immediate shutdown")
    
    for process in current_processes:
        try:
            process.send_signal(signal.SIGINT)
        except:
            pass
    
    start_wait = time.time()
    all_stopped = False
    
    while time.time() - start_wait < 10:
        all_stopped = True
        for process in current_processes:
            if process.poll() is None:
                all_stopped = False
                break
        
        if all_stopped:
            break
        time.sleep(0.5)
    
    if not all_stopped:
        print("[IRIS] Forcing shutdown...")
        for process in current_processes:
            try:
                process.kill()
            except:
                pass
    
    print("[IRIS] All services stopped")
    sys.exit(0)

def print_banner(mode):
    """Print I.R.I.S. startup banner"""
    mode_labels = {
        'api': 'API Only',
        'html': 'HTML Frontend',
        'react': 'React Frontend',
        'full': 'Full (HTML + React)'
    }
    banner = f"""
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║              I.R.I.S. v1.2.0                     ║
    ║   Intelligent Rendering & Image Synthesis        ║
    ║                                                  ║
    ║   Mode: {mode_labels.get(mode, mode):<40} ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)

def load_settings():
    """Load settings from settings.json"""
    project_root = Path(__file__).resolve().parents[1]
    settings_path = project_root / "settings.json"
    
    default_settings = {
        "discordEnabled": False,
        "dramEnabled": False,
        "vramThreshold": 6,
        "maxDram": 8,
        "nsfwStrength": 2
    }
    
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                return {**default_settings, **settings}
        except Exception as e:
            print(f"[WARN] Could not load settings.json: {e}")
    
    return default_settings

def start_web_server(mode='html'):
    """Start FastAPI Web UI Server with specified mode"""
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    
    # Set environment variables based on mode
    env = os.environ.copy()
    env["IRIS_MODE"] = mode
    
    # Determine what to enable
    enable_html = mode in ('html', 'full')
    enable_react = mode in ('react', 'full')
    
    if not enable_html:
        env["IRIS_NO_HTML"] = "1"
    if enable_react:
        env["IRIS_SERVE_REACT"] = "1"
    
    print("\n[WEB] Starting I.R.I.S. Server...")
    print(f"      Mode: {mode}")
    print(f"      API:   http://localhost:8000/api")
    
    if enable_html:
        print(f"      HTML:  http://localhost:8000")
    if enable_react:
        react_dist = project_root / "frontend-react" / "dist"
        if react_dist.exists():
            print(f"      React: http://localhost:8000/app")
        else:
            print(f"      React: [!] Build missing - run 'npm run build' in frontend-react/")
    
    if mode == 'api':
        print(f"      [No frontend - API only mode]")
    
    print()

    process = subprocess.Popen([
        sys.executable,
        "-m", "uvicorn",
        "src.api.server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ], env=env)

    current_processes.append(process)
    return process

def start_discord_bot():
    """Start Discord Bot with Rich Presence"""
    print("[BOT] Starting Discord Bot...")
    print("      Rich Presence will show generation status\n")
    
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    
    process = subprocess.Popen([sys.executable, "src/services/bot.py"])
    current_processes.append(process)
    return process

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='I.R.I.S. - Intelligent Rendering & Image Synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  api     Only API server, no frontend (for development with external frontend)
  html    API + classic HTML frontend (default)
  react   API + React production build (served from frontend-react/dist)
  full    API + HTML + React (both frontends available)

Examples:
  python src/start.py                  # Start with HTML frontend
  python src/start.py --mode api       # API only for React dev server
  python src/start.py --mode react     # Serve React production build
  python src/start.py --mode full      # Both frontends
        """
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['api', 'html', 'react', 'full'],
        default='html',
        help='Server mode (default: html)'
    )
    parser.add_argument(
        '--no-bot',
        action='store_true',
        help='Disable Discord bot even if enabled in settings'
    )
    return parser.parse_args()

def main():
    """Main entry point"""
    global restart_requested, shutdown_in_progress
    
    args = parse_args()
    
    print_banner(args.mode)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load settings
    settings = load_settings()
    discord_enabled = settings.get("discordEnabled", False) and not args.no_bot
    
    # Show startup info
    print(f"    [CONFIG] Mode: {args.mode}")
    print(f"    [CONFIG] Discord Bot: {'Enabled' if discord_enabled else 'Disabled'}")
    print(f"    [CONFIG] DRAM Extension: {'Enabled' if settings.get('dramEnabled') else 'Disabled'}")
    print()
    print("    [TIP] Press CTRL+C to exit | CTRL+R to restart")
    print()
    
    # Start keyboard listener for CTRL+R (Windows only)
    if sys.platform == 'win32':
        def keyboard_listener():
            global restart_requested, shutdown_in_progress
            try:
                while not shutdown_in_progress:
                    try:
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            # CTRL+R = \x12 (18 decimal)
                            if key == b'\x12':
                                print("\n[IRIS] Restart requested (CTRL+R)...")
                                restart_requested = True
                                # Terminate processes
                                for process in current_processes:
                                    try:
                                        # Use taskkill for reliable termination on Windows
                                        subprocess.run(
                                            ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                            capture_output=True,
                                            timeout=5
                                        )
                                    except:
                                        try:
                                            process.terminate()
                                        except:
                                            pass
                                break
                    except Exception:
                        pass
                    time.sleep(0.05)
            except Exception as e:
                print(f"[WARN] Keyboard listener error: {e}")
        
        kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
        kb_thread.start()
    
    while True:
        restart_requested = False
        current_processes.clear()
        shutdown_in_progress = False
        
        # Clean up any old restart signal file
        project_root = Path(__file__).resolve().parents[1]
        signal_file = project_root / "restart_signal"
        if signal_file.exists():
            signal_file.unlink()
        
        try:
            # Start web server with specified mode
            web_process = start_web_server(mode=args.mode)
            
            # Start Discord bot if enabled
            bot_process = None
            if discord_enabled:
                time.sleep(0.5)
                bot_process = start_discord_bot()
            
            # Wait for processes - check periodically for restart
            while True:
                # Check if restart was requested via CTRL+R
                if restart_requested:
                    break
                
                # Check if restart was requested via Admin Panel (signal file)
                if signal_file.exists():
                    print("\n[IRIS] Restart requested via Admin Panel...")
                    signal_file.unlink()
                    restart_requested = True
                    break
                
                # Check if web process ended
                if web_process.poll() is not None:
                    break
                    
                time.sleep(0.2)
            
            # If restart requested, clean up and continue
            if restart_requested:
                print("[IRIS] Restarting services...")
                # Make sure all processes are stopped
                for process in current_processes:
                    if process.poll() is None:
                        try:
                            if sys.platform == 'win32':
                                subprocess.run(
                                    ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                    capture_output=True,
                                    timeout=5
                                )
                            else:
                                process.terminate()
                        except:
                            pass
                
                # Wait for processes to end
                for process in current_processes:
                    try:
                        process.wait(timeout=3)
                    except:
                        pass
                
                time.sleep(1)
                print("[IRIS] Starting fresh...\n")
                continue
            else:
                break
                
        except KeyboardInterrupt:
            if restart_requested:
                continue
            break

if __name__ == "__main__":
    main()
