import discord
from discord.ext import commands, tasks
import os
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import json
from dotenv import load_dotenv

# Calculate path to .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / '.env'

# Load .env file
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

sys.path.insert(0, str(BASE_DIR))
from src.utils.logger import create_logger

logger = create_logger("IRISDiscordBot")

# Paths for data
SENT_IMAGES_FILE = BASE_DIR / "static" / "data" / "img_send.json"
PROMPTS_LOG_FILE = BASE_DIR / "static" / "data" / "prompts_history.json"
BOT_STATUS_FILE = BASE_DIR / "static" / "data" / "bot_status.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Configuration functions
def read_config_file(filename):
    """Read configuration from a file in static/config folder"""
    filepath = BASE_DIR / "static" / "config" / filename
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"{filepath} is empty!")
                return None
            return content
    except FileNotFoundError:
        logger.error(f"{filepath} not found!")
        return None

def get_env_id(key, default=0):
    """Helper function to safely load IDs as integers from .env"""
    val = os.getenv(key)
    if not val:
        logger.warning(f"Variable {key} not found in .env. Using default: {default}")
        return default
    try:
        return int(val)
    except ValueError:
        logger.error(f"Invalid ID for {key} in .env (must be a number)!")
        return default

# Bot Configuration
logger.info("Loading Discord Bot Configuration...")
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN') or read_config_file("bot_token.txt")
BOT_OWNER_ID = os.getenv('DISCORD_BOT_OWNER_ID') or read_config_file("bot_owner_id.txt")
BOT_ID = os.getenv('DISCORD_BOT_ID') or read_config_file("bot_id.txt")

CHANNEL_NEW_IMAGES = get_env_id('DISCORD_CHANNEL_NEW_IMAGES')
CHANNEL_VARIATIONS = get_env_id('DISCORD_CHANNEL_VARIATIONS')
CHANNEL_UPSCALED = get_env_id('DISCORD_CHANNEL_UPSCALED')

# Image Tracking Functions
sent_images_dict = {}
currently_processing = set()

def load_sent_images():
    """Load sent images from JSON file"""
    global sent_images_dict
    if SENT_IMAGES_FILE.exists():
        try:
            with open(SENT_IMAGES_FILE, 'r', encoding='utf-8') as f:
                sent_images_dict = json.load(f)
            logger.info(f"{len(sent_images_dict)} previously sent images loaded")
        except Exception as e:
            logger.error(f"Error loading sent images: {e}")
            sent_images_dict = {}

def save_sent_image(filename, message_link):
    """Save sent image to JSON file"""
    sent_images_dict[filename] = {
        "message_link": message_link,
        "sent_at": datetime.now().isoformat()
    }
    try:
        SENT_IMAGES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SENT_IMAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(sent_images_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving sent image: {e}")

# Bot Setup
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

monitor_lock = asyncio.Lock()

# Status tracking
current_status = "idle"
generation_count = 0

async def update_bot_status(status: str, details: str = None):
    """Update the bot's Discord status/activity"""
    global current_status
    current_status = status
    
    if status == "idle":
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="for new images 👀"
        )
    elif status == "monitoring":
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name=f"outputs folder | {generation_count} sent"
        )
    elif status == "sending":
        activity = discord.Activity(
            type=discord.ActivityType.playing,
            name=f"Sending: {details}" if details else "Sending image..."
        )
    elif status == "generating":
        activity = discord.Activity(
            type=discord.ActivityType.playing,
            name=f"🎨 Generating..."
        )
    elif status == "restarting":
        activity = discord.Activity(
            type=discord.ActivityType.playing,
            name="🔄 Server restarting..."
        )
    else:
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="I.R.I.S. Server"
        )
    
    try:
        await bot.change_presence(activity=activity, status=discord.Status.online)
    except Exception as e:
        logger.error(f"Failed to update status: {e}")

@bot.event
async def on_ready():
    logger.discord_bot_ready(bot.user.name)
    logger.info("=" * 70)
    logger.info(f"Connected as: {bot.user.name}")
    
    # Set initial status
    await update_bot_status("idle")
    
    channels = {
        "New": (bot.get_channel(CHANNEL_NEW_IMAGES), CHANNEL_NEW_IMAGES),
        "Variations": (bot.get_channel(CHANNEL_VARIATIONS), CHANNEL_VARIATIONS),
        "Upscaled": (bot.get_channel(CHANNEL_UPSCALED), CHANNEL_UPSCALED)
    }

    all_found = True
    for name, (channel, channel_id) in channels.items():
        if channel:
            logger.success(f"{name} Channel found: #{channel.name}")
        else:
            logger.error(f"{name} Channel ID {channel_id} invalid!")
            all_found = False

    if all_found:
        load_sent_images()
        if not monitor_images.is_running():
            monitor_images.start()
            logger.success("Image monitoring started")
        if not monitor_server_status.is_running():
            monitor_server_status.start()
            logger.success("Server status monitoring started")
        await update_bot_status("monitoring")
    else:
        logger.error("MONITORING NOT STARTED: Please check the .env file.")

@tasks.loop(seconds=2.0)
async def monitor_server_status():
    """Monitor server status file for generation updates"""
    try:
        if not BOT_STATUS_FILE.exists():
            return
        
        with open(BOT_STATUS_FILE, 'r') as f:
            status_data = json.load(f)
        
        server_status = status_data.get("status", "idle")
        details = status_data.get("details", "")
        
        global current_status
        if server_status == "generating" and current_status != "generating":
            await update_bot_status("generating", details)
        elif server_status == "idle" and current_status == "generating":
            await update_bot_status("monitoring")
        elif server_status == "restarting":
            await update_bot_status("restarting")
            
    except Exception as e:
        pass  # Ignore errors reading status file

@tasks.loop(seconds=3.0)
async def monitor_images():
    async with monitor_lock:
        try:
            if not OUTPUTS_DIR.exists(): return
            
            chan_new = bot.get_channel(CHANNEL_NEW_IMAGES)
            chan_var = bot.get_channel(CHANNEL_VARIATIONS)
            chan_up = bot.get_channel(CHANNEL_UPSCALED)
            
            if not all([chan_new, chan_var, chan_up]): return

            image_files = sorted(OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
            
            for image_path in image_files:
                filename = image_path.name
                if filename in sent_images_dict or filename in currently_processing:
                    continue
                
                currently_processing.add(filename)
                
                await asyncio.sleep(12)
                
                if filename in sent_images_dict:
                    currently_processing.discard(filename)
                    continue
                
                is_up = filename.startswith(("up", "upscaled"))
                is_var = filename.startswith(("var", "variation"))
                target_channel = chan_up if is_up else (chan_var if is_var else chan_new)

                try:
                    if image_path.stat().st_size > 8 * 1024 * 1024:
                        logger.warning(f"Image too large for Discord: {filename}")
                        currently_processing.discard(filename)
                        continue

                    seed = "Unknown"
                    parts = filename.replace(".png", "").split("_")
                    if len(parts) >= 4 and parts[3].isdigit(): seed = parts[3]

                    embed = discord.Embed(
                        title="I.R.I.S. Rendering",
                        description=f"**Seed:** `{seed}`",
                        color=0x14b8a6 if is_up else 0x06b6d4,
                        timestamp=datetime.now()
                    )
                    embed.set_footer(text="I.R.I.S. - Intelligent Rendering & Image Synthesis")
                    
                    # Update status while sending
                    await update_bot_status("sending", filename)
                    
                    with open(image_path, 'rb') as f:
                        df = discord.File(f, filename=filename)
                        embed.set_image(url=f"attachment://{filename}")
                        msg = await target_channel.send(file=df, embed=embed)
                        
                        from src.utils.file_manager import FileManager
                        FileManager.log_sent_image(filename, msg.jump_url)
                        
                        save_sent_image(filename, msg.jump_url)
                        logger.success(f"Image sent: {filename}")
                        
                        # Update generation count and status
                        global generation_count
                        generation_count += 1
                        await update_bot_status("monitoring")

                except Exception as e:
                    logger.error(f"Error sending {filename}: {e}")
                finally:
                    currently_processing.discard(filename)
        except Exception as e:
            logger.error(f"Monitor error: {e}")

@bot.command(name='iris')
async def iris_info(ctx):
    embed = discord.Embed(title="I.R.I.S. Bot Status", color=0x06b6d4)
    embed.add_field(name="📊 Sent (Session)", value=str(generation_count), inline=True)
    embed.add_field(name="📁 Total Tracked", value=str(len(sent_images_dict)), inline=True)
    embed.add_field(name="🔄 Status", value=current_status.capitalize(), inline=True)
    embed.set_footer(text="I.R.I.S. - Intelligent Rendering & Image Synthesis")
    await ctx.send(embed=embed)

@bot.command(name='cleanup')
async def cleanup_channels(ctx):
    """Remove Discord messages for images that no longer exist in outputs/ folder"""
    try:
        # Permission check
        owner_id = str(BOT_OWNER_ID).strip() if BOT_OWNER_ID else None
        author_id = str(ctx.author.id)
        
        logger.info(f"Cleanup requested by {ctx.author.name} (ID: {author_id})")
        
        if not owner_id:
            await ctx.send("❌ BOT_OWNER_ID nicht konfiguriert in .env")
            return
        
        if author_id != owner_id:
            await ctx.send(f"❌ Du hast keine Berechtigung für diesen Command.")
            return
        
        # Get existing files in outputs/
        existing_files = set()
        if OUTPUTS_DIR.exists():
            existing_files = {f.name for f in OUTPUTS_DIR.glob("*.png")}
        
        logger.info(f"Found {len(existing_files)} images in outputs/")
        
        # Get channels
        channels = []
        if CHANNEL_NEW_IMAGES:
            ch = bot.get_channel(CHANNEL_NEW_IMAGES)
            if ch: channels.append((ch, "New Images"))
        if CHANNEL_VARIATIONS:
            ch = bot.get_channel(CHANNEL_VARIATIONS)
            if ch: channels.append((ch, "Variations"))
        if CHANNEL_UPSCALED:
            ch = bot.get_channel(CHANNEL_UPSCALED)
            if ch: channels.append((ch, "Upscaled"))
        
        if not channels:
            await ctx.send("❌ Keine Channels gefunden!")
            return
        
        # Status message
        embed = discord.Embed(
            title="🧹 Channel Cleanup",
            description=f"Scanne {len(channels)} Channels...\n\nLösche Nachrichten für Bilder die nicht mehr in `outputs/` existieren.",
            color=0xfbbf24
        )
        status_msg = await ctx.send(embed=embed)
        
        to_delete = []
        scanned_count = 0
        
        for channel, channel_name in channels:
            permissions = channel.permissions_for(channel.guild.me)
            if not permissions.read_message_history or not permissions.manage_messages:
                logger.warning(f"Missing permissions in {channel_name}")
                continue
            
            logger.info(f"Scanning {channel_name}...")
            
            try:
                async for message in channel.history(limit=1000):
                    scanned_count += 1
                    
                    # Only check bot messages with attachments
                    if message.author.id != bot.user.id:
                        continue
                    if not message.attachments:
                        continue
                    
                    # Get filename from attachment
                    filename = message.attachments[0].filename
                    
                    # Check if file still exists in outputs/
                    if filename not in existing_files:
                        to_delete.append((message, filename, channel_name))
                        
            except Exception as e:
                logger.error(f"Error scanning {channel_name}: {e}")
        
        logger.info(f"Scanned {scanned_count} messages, found {len(to_delete)} to delete")
        
        if not to_delete:
            embed = discord.Embed(
                title="✅ Cleanup Abgeschlossen",
                description=f"Keine verwaisten Nachrichten gefunden.\n\n"
                           f"📊 {scanned_count} Nachrichten gescannt\n"
                           f"📁 {len(existing_files)} Bilder in outputs/",
                color=0x10b981
            )
            await status_msg.edit(embed=embed)
            return
        
        # Show what will be deleted
        preview = "\n".join([f"• {fn}" for _, fn, _ in to_delete[:10]])
        if len(to_delete) > 10:
            preview += f"\n... und {len(to_delete) - 10} weitere"
        
        embed = discord.Embed(
            title="⚠️ Bestätigung erforderlich",
            description=f"**{len(to_delete)} Nachrichten** werden gelöscht:\n\n{preview}\n\n"
                       f"Reagiere mit ✅ zum Bestätigen oder ❌ zum Abbrechen.",
            color=0xef4444
        )
        await status_msg.edit(embed=embed)
        await status_msg.add_reaction('✅')
        await status_msg.add_reaction('❌')
        
        def check(reaction, user):
            return user == ctx.author and str(reaction.emoji) in ['✅', '❌'] and reaction.message.id == status_msg.id
        
        try:
            reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=check)
            
            if str(reaction.emoji) == '❌':
                embed = discord.Embed(
                    title="❌ Cleanup Abgebrochen",
                    description="Keine Nachrichten wurden gelöscht.",
                    color=0x6b7280
                )
                await status_msg.edit(embed=embed)
                try: await status_msg.clear_reactions()
                except: pass
                return
        
        except asyncio.TimeoutError:
            embed = discord.Embed(
                title="⏱️ Timeout",
                description="Keine Reaktion erhalten. Cleanup abgebrochen.",
                color=0x6b7280
            )
            await status_msg.edit(embed=embed)
            try: await status_msg.clear_reactions()
            except: pass
            return
        
        # Delete messages
        try: await status_msg.clear_reactions()
        except: pass
        
        total_deleted = 0
        failed = 0
        
        for i, (message, filename, channel_name) in enumerate(to_delete, 1):
            try:
                if i % 5 == 0 or i == len(to_delete):
                    embed = discord.Embed(
                        title="🧹 Cleanup läuft...",
                        description=f"Fortschritt: {i}/{len(to_delete)} ({int(i/len(to_delete)*100)}%)\n"
                                  f"✅ Gelöscht: {total_deleted}\n"
                                  f"❌ Fehler: {failed}",
                        color=0x3b82f6
                    )
                    await status_msg.edit(embed=embed)
                
                await message.delete()
                total_deleted += 1
                
                # Also remove from img_send.json
                if filename in sent_images_dict:
                    del sent_images_dict[filename]
                
                logger.info(f"Gelöscht: {filename}")
                await asyncio.sleep(1.0)  # Rate limit
                
            except discord.errors.NotFound:
                failed += 1
            except discord.errors.Forbidden:
                logger.error(f"Keine Berechtigung: {filename}")
                failed += 1
            except Exception as e:
                logger.error(f"Fehler: {filename} - {e}")
                failed += 1
        
        # Save updated img_send.json
        try:
            with open(SENT_IMAGES_FILE, 'w', encoding='utf-8') as f:
                json.dump(sent_images_dict, f, indent=2, ensure_ascii=False)
        except: pass
        
        embed = discord.Embed(
            title="✅ Cleanup Abgeschlossen",
            description=f"**Ergebnis:**\n"
                       f"✅ Gelöscht: {total_deleted}\n"
                       f"❌ Fehler: {failed}\n"
                       f"📊 img_send.json aktualisiert",
            color=0x10b981
        )
        await status_msg.edit(embed=embed)
        logger.success(f"Cleanup: {total_deleted} gelöscht, {failed} Fehler")
        
    except Exception as e:
        logger.error(f"Cleanup Fehler: {e}")
        import traceback
        traceback.print_exc()
        try:
            await ctx.send(f"❌ Cleanup Fehler: {str(e)}")
        except:
            pass

@bot.command(name='help')
async def help_command(ctx):
    embed = discord.Embed(title="I.R.I.S. Help", color=0x06b6d4)
    embed.add_field(name="!iris", value="Show bot status", inline=False)
    embed.add_field(name="!cleanup", value="Remove images not in img_send.json (Owner only)", inline=False)
    embed.add_field(name="!debug", value="Show bot configuration (Owner only)", inline=False)
    embed.add_field(name="!help", value="Show this help message", inline=False)
    await ctx.send(embed=embed)


@bot.command(name='debug')
async def debug_info(ctx):
    """Show bot configuration for debugging (Owner only)"""
    owner_id = str(BOT_OWNER_ID).strip() if BOT_OWNER_ID else None
    author_id = str(ctx.author.id)
    
    # Only owner can see debug info
    if owner_id and author_id != owner_id:
        await ctx.send("❌ Nur der Bot-Owner kann diesen Command nutzen.")
        return
    
    # Check channels
    ch_new = bot.get_channel(CHANNEL_NEW_IMAGES)
    ch_var = bot.get_channel(CHANNEL_VARIATIONS)
    ch_up = bot.get_channel(CHANNEL_UPSCALED)
    
    # Load img_send.json stats
    load_sent_images()
    
    embed = discord.Embed(title="🔧 I.R.I.S. Debug Info", color=0x8b5cf6)
    
    # IDs
    embed.add_field(
        name="📋 Konfiguration",
        value=f"**Bot ID:** `{BOT_ID or 'Nicht gesetzt'}`\n"
              f"**Owner ID:** `{owner_id or 'Nicht gesetzt'}`\n"
              f"**Deine ID:** `{author_id}`\n"
              f"**Owner Match:** {'✅' if author_id == owner_id else '❌'}",
        inline=False
    )
    
    # Channels
    embed.add_field(
        name="📺 Channels",
        value=f"**New Images:** {f'#{ch_new.name}' if ch_new else f'❌ ID {CHANNEL_NEW_IMAGES}'}\n"
              f"**Variations:** {f'#{ch_var.name}' if ch_var else f'❌ ID {CHANNEL_VARIATIONS}'}\n"
              f"**Upscaled:** {f'#{ch_up.name}' if ch_up else f'❌ ID {CHANNEL_UPSCALED}'}",
        inline=False
    )
    
    # Files
    embed.add_field(
        name="📁 Dateien",
        value=f"**img_send.json:** {len(sent_images_dict)} Einträge\n"
              f"**outputs/:** {len(list(OUTPUTS_DIR.glob('*.png')))} Bilder",
        inline=False
    )
    
    # Permissions check
    if ch_new:
        perms = ch_new.permissions_for(ch_new.guild.me)
        embed.add_field(
            name="🔐 Berechtigungen (New Images)",
            value=f"**Send Messages:** {'✅' if perms.send_messages else '❌'}\n"
                  f"**Attach Files:** {'✅' if perms.attach_files else '❌'}\n"
                  f"**Read History:** {'✅' if perms.read_message_history else '❌'}\n"
                  f"**Manage Messages:** {'✅' if perms.manage_messages else '❌'}\n"
                  f"**Add Reactions:** {'✅' if perms.add_reactions else '❌'}",
            inline=False
        )
    
    await ctx.send(embed=embed)

def main():
    if not BOT_TOKEN:
        logger.error("No Bot Token found!")
        return
    try:
        import signal
        
        def shutdown_handler(signum, frame):
            logger.info("Received shutdown signal, closing bot...")
            try:
                if bot.loop and bot.loop.is_running():
                    asyncio.run_coroutine_threadsafe(bot.close(), bot.loop)
                else:
                    sys.exit(0)
            except:
                sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        if os.name == 'nt':
            try:
                signal.signal(signal.SIGBREAK, shutdown_handler)
            except:
                pass
        
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except SystemExit:
        logger.info("Bot shutdown complete")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Bot process ending")

if __name__ == "__main__":
    main()
