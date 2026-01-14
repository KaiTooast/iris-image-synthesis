import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
from logging.handlers import RotatingFileHandler


# ============ Configuration ============

# Log level from environment variable (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.environ.get("IRIS_LOG_LEVEL", "INFO").upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Session log retention
SESSION_LOG_KEEP_COUNT = 10  # Keep last 10 session logs
SESSION_LOG_MAX_AGE_DAYS = 7  # Delete logs older than 7 days


# ANSI Color Codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.WHITE,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }
    
    def format(self, record):
        # Get color for level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format message based on content
        message = record.getMessage()
        
        # Special formatting for different message types
        if message.startswith("[SUCCESS]"):
            message = message.replace("[SUCCESS]", "")
            prefix = f"{Colors.BRIGHT_GREEN}✓{Colors.RESET}"
            return f"{Colors.DIM}{timestamp}{Colors.RESET} {prefix} {Colors.GREEN}{message.strip()}{Colors.RESET}"
        
        elif message.startswith("[ERROR]"):
            message = message.replace("[ERROR]", "")
            prefix = f"{Colors.BRIGHT_RED}✗{Colors.RESET}"
            return f"{Colors.DIM}{timestamp}{Colors.RESET} {prefix} {Colors.RED}{message.strip()}{Colors.RESET}"
        
        elif message.startswith("[WARNING]"):
            message = message.replace("[WARNING]", "")
            prefix = f"{Colors.YELLOW}⚠{Colors.RESET}"
            return f"{Colors.DIM}{timestamp}{Colors.RESET} {prefix} {Colors.YELLOW}{message.strip()}{Colors.RESET}"
        
        elif "═" in message or "─" in message:
            # Separator lines
            return f"{Colors.DIM}{message}{Colors.RESET}"
        
        elif message.startswith("   "):
            # Indented detail lines
            return f"{Colors.DIM}{timestamp}{Colors.RESET}   {Colors.DIM}{message.strip()}{Colors.RESET}"
        
        else:
            # Regular messages
            level_indicator = {
                logging.DEBUG: f"{Colors.DIM}DBG{Colors.RESET}",
                logging.INFO: f"{Colors.CYAN}INF{Colors.RESET}",
                logging.WARNING: f"{Colors.YELLOW}WRN{Colors.RESET}",
                logging.ERROR: f"{Colors.RED}ERR{Colors.RESET}",
            }.get(record.levelno, "   ")
            
            return f"{Colors.DIM}{timestamp}{Colors.RESET} {level_indicator} {message}"


class AIGeneratorLogger:
    """
    Custom logger for AI Image Generator
    Uses rotating log files to prevent disk space issues
    
    Environment Variables:
        IRIS_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR)
    """
    
    def __init__(self, name: str = "AIGenerator", log_session: bool = True):
        self.name = name
        self.logs_dir = Path("Logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Get configured log level
        self.log_level = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)
        
        # Main log file with rotation
        self.log_file = self.logs_dir / "iris.log"
        
        # Session-specific log for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.logs_dir / f"session_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Rotating file handler (main log - keeps last 5 files, 10MB each)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        rotating_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        rotating_handler.setLevel(logging.DEBUG)  # File gets everything
        rotating_handler.setFormatter(file_formatter)
        
        # Session file handler (detailed logs for current session)
        session_handler = logging.FileHandler(
            self.session_log_file, 
            encoding='utf-8'
        )
        session_handler.setLevel(logging.DEBUG)
        session_handler.setFormatter(file_formatter)
        
        # Console handler (user-friendly colored output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)  # Respect configured level
        console_handler.setFormatter(ColoredFormatter())
        
        # Fix encoding issues on Windows
        if sys.platform == 'win32':
            # Enable ANSI colors on Windows
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                pass
        
        if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
            try:
                console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            except:
                pass
        
        # Add handlers
        self.logger.addHandler(rotating_handler)
        self.logger.addHandler(session_handler)
        self.logger.addHandler(console_handler)
        
        # Session start
        self.session_start_time = datetime.now()
        if log_session:
            self.log_session_start()
        
        # Clean up old session logs
        self._cleanup_old_sessions()
    
    def log_session_start(self):
        """Log session start with system info"""
        self.logger.info("═" * 60)
        self.logger.info(f"  I.R.I.S. - AI Image Generator")
        self.logger.info(f"  Session: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("═" * 60)
    
    def _cleanup_old_sessions(self):
        """Remove old session log files based on count and age"""
        try:
            session_files = sorted(
                self.logs_dir.glob("session_*.log"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            cutoff_date = datetime.now() - timedelta(days=SESSION_LOG_MAX_AGE_DAYS)
            
            for i, log_file in enumerate(session_files):
                # Keep the most recent ones regardless of age
                if i < SESSION_LOG_KEEP_COUNT:
                    continue
                
                # Delete if older than max age or exceeds count
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date or i >= SESSION_LOG_KEEP_COUNT:
                        log_file.unlink()
                except Exception:
                    pass
        except Exception:
            pass
    
    def log_session_end(self):
        """Log session end with statistics"""
        session_end_time = datetime.now()
        duration = session_end_time - self.session_start_time
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        seconds = duration.seconds % 60
        
        self.logger.info("")
        self.logger.info("═" * 60)
        self.logger.info(f"  Session Ended - Duration: {hours}h {minutes}m {seconds}s")
        self.logger.info("═" * 60)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[WARNING] {message}")
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(f"[ERROR] {message}")
    
    def success(self, message: str):
        """Log success message"""
        self.logger.info(f"[SUCCESS] {message}")
    
    def generation_start(self, prompt: str, settings: dict):
        """Log image generation start"""
        self.logger.info("")
        self.logger.info("─" * 50)
        self.logger.info("Generation Started")
        self.logger.info(f"   Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        self.logger.info(f"   Size: {settings.get('width')}×{settings.get('height')} | Steps: {settings.get('steps')} | CFG: {settings.get('cfg_scale')}")
        self.logger.info(f"   Seed: {settings.get('seed', 'Random')}")
    
    def generation_progress(self, step: int, total_steps: int, eta: float):
        """Log generation progress"""
        progress = (step / total_steps) * 100
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)
        self.logger.debug(
            f"   Progress: Step {step}/{total_steps} ({progress:.1f}%) - "
            f"ETA: {eta_min}m {eta_sec}s"
        )
    
    def generation_complete(self, filename: str, generation_time: float, seed: int):
        """Log generation completion"""
        self.logger.info(f"[SUCCESS] Generated: {filename}")
        self.logger.info(f"   Time: {generation_time:.1f}s | Seed: {seed}")
        self.logger.info("─" * 50)
    
    def gpu_info(self, gpu_name: str, vram_total: float, vram_used: float, temp: float):
        """Log GPU information"""
        vram_perc = (vram_used / vram_total * 100) if vram_total > 0 else 0
        temp_icon = "🔥" if temp > 80 else "🌡️" if temp > 60 else "❄️"
        self.logger.info(f"GPU: {gpu_name}")
        self.logger.info(f"   VRAM: {vram_used:.1f}/{vram_total:.1f}GB ({vram_perc:.0f}%) | Temp: {temp:.0f}°C {temp_icon}")
    
    def upscale_start(self, filename: str, scale: int, method: str = "lanczos"):
        """Log upscale operation start"""
        self.logger.info(f"Upscaling: {filename} ({scale}x) using {method}")
    
    def upscale_complete(self, new_filename: str, new_size: tuple):
        """Log upscale completion"""
        self.logger.info(f"[SUCCESS] Upscaled: {new_filename} ({new_size[0]}x{new_size[1]})")
    
    def variation_start(self, filename: str, strength: float):
        """Log variation creation start"""
        self.logger.info(f"Creating Variation: {filename} (strength: {strength})")
    
    def variation_complete(self, new_filename: str, generation_time: float):
        """Log variation completion"""
        self.logger.info(f"[SUCCESS] Variation Complete: {new_filename} ({generation_time:.1f}s)")
    
    def websocket_connect(self):
        """Log WebSocket connection"""
        self.logger.debug("WebSocket Client Connected")
    
    def websocket_disconnect(self):
        """Log WebSocket disconnection"""
        self.logger.debug("WebSocket Client Disconnected")
    
    def model_load_start(self, model_name: str):
        """Log model loading start"""
        self.logger.info(f"Loading Model: {model_name}")
        self.logger.info("This may take 5-10 minutes on first run...")
    
    def model_load_complete(self, model_name: str, device: str):
        """Log model loading completion"""
        self.logger.info(f"[SUCCESS] Model Loaded: {model_name} on {device}")
    
    def nsfw_blocked(self, reason: str):
        """Log NSFW content block"""
        self.logger.warning(f"Content Blocked: {reason}")
    
    def discord_bot_start(self):
        """Log Discord bot start"""
        self.logger.info("Discord Bot Starting...")
    
    def discord_bot_ready(self, bot_name: str):
        """Log Discord bot ready"""
        self.logger.info(f"[SUCCESS] Discord Bot Ready: {bot_name}")
    
    def discord_image_sent(self, filename: str, channel: str):
        """Log image sent to Discord"""
        self.logger.debug(f"Sent to Discord: {filename} -> #{channel}")
    
    def cleanup(self):
        """Clean up and close logger"""
        self.log_session_end()
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# Singleton cache for loggers
_logger_instances = {}
_session_logged = False

def create_logger(name: str = "AIGenerator") -> AIGeneratorLogger:
    """
    Factory function to create a logger instance (singleton per name)
    Usage: logger = create_logger()
    """
    global _session_logged
    if name not in _logger_instances:
        _logger_instances[name] = AIGeneratorLogger(name, log_session=not _session_logged)
        if not _session_logged:
            _session_logged = True
    return _logger_instances[name]


# Example usage and testing
if __name__ == "__main__":
    # Create logger
    logger = create_logger()
    
    # Test various log types
    logger.info("Testing logger functionality")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.success("This is a success message")
    
    # Test generation logs
    logger.generation_start(
        "anime girl with blue hair", 
        {"width": 512, "height": 768, "steps": 30, "cfg_scale": 10, "seed": 12345}
    )
    logger.generation_progress(15, 30, 45.5)
    logger.generation_complete("test_image.png", 90.5, 12345)
    
    # Test GPU info
    logger.gpu_info("NVIDIA GTX 1650", 4.0, 2.5, 65)
    
    # End session
    logger.cleanup()
