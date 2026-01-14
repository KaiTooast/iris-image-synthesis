# 📖 Setup Guide
### Complete Installation Guide for I.R.I.S. v1.2.0

*Intelligent Rendering & Image Synthesis*

---

## 💻 System Requirements

### Hardware

| Tier | GPU | VRAM | RAM | Storage | Performance |
|------|-----|------|-----|---------|-------------|
| **Minimum** | GTX 1650 | 4GB | 8GB | 20GB | ~90 sec / 512×768 |
| **Recommended** | RTX 3060 | 12GB | 16GB | 50GB | ~30 sec / 512×768 |
| **High-End** | RTX 4090 | 24GB | 32GB | 100GB | ~10 sec / 512×768 |
| **CPU-Only** | 8+ cores | N/A | 16GB+ | 20GB | 10-30 min / image |

### Supported GPUs

| Vendor | Technology | Status |
|--------|------------|--------|
| NVIDIA | CUDA | ✅ Full Support |
| AMD | ROCm | 🧪 Experimental |
| Intel | oneAPI (XPU) | 🧪 Experimental |
| Apple | Metal (MPS) | 🧪 Experimental |
| CPU | PyTorch CPU | ✅ Always Available |

### Software

**Required:**
- Python 3.9, 3.10, or 3.11
- Git

**Optional:**
- CUDA 11.8+ (NVIDIA)
- ROCm 5.6+ (AMD)
- oneAPI (Intel Arc)
- Node.js 18+ (for React frontend)
- Discord Account (for bot integration)

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/KaiTooast/iris.git
cd iris
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install React Frontend

```bash
cd frontend-react
npm install
cd ..
```

### 5. (Optional) Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

---

## 🎮 PyTorch Installation by GPU

### NVIDIA (CUDA)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### AMD (ROCm)

```bash
# ROCm 5.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# ROCm 6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Intel Arc (oneAPI)

```bash
pip install torch torchvision torchaudio intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### CPU Only

```bash
pip install torch torchvision torchaudio
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 🎬 Running I.R.I.S.

### Start Server

```bash
# Windows (recommended)
.\venv\Scripts\python.exe src/start.py

# Linux / macOS
python src/start.py

# Without Discord bot
python src/start.py --no-bot
```

### Access Web UI

| Frontend | URL | Notes |
|----------|-----|-------|
| HTML | http://localhost:8000 | Ready immediately |
| React | http://localhost:3000 | Requires `npm run dev` |

### Start React Frontend (Optional)

```bash
cd frontend-react
npm run dev
```

---

## ⚙️ Configuration

### settings.json

```json
{
  "dramEnabled": true,
  "vramThreshold": 6,
  "maxDram": 16,
  "nsfwEnabled": true,
  "nsfwStrength": 2,
  "discordEnabled": false
}
```

| Setting | Type | Description |
|---------|------|-------------|
| `dramEnabled` | bool | Use system RAM when VRAM is low |
| `vramThreshold` | int | VRAM threshold (GB) to enable DRAM Extension |
| `maxDram` | int | Maximum system RAM to use (GB) |
| `nsfwEnabled` | bool | Enable NSFW prompt filter |
| `nsfwStrength` | int | 1=Minimal, 2=Standard, 3=Strict |
| `discordEnabled` | bool | Auto-start Discord bot |

### .env (Optional)

```env
HOST=0.0.0.0
PORT=8000
DEFAULT_MODEL=anime_kawai

# Discord Bot
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_BOT_ID=your_bot_id
DISCORD_BOT_OWNER_ID=your_user_id

DISCORD_CHANNEL_NEW_IMAGES=channel_id
DISCORD_CHANNEL_VARIATIONS=channel_id
DISCORD_CHANNEL_UPSCALED=channel_id
```

---

## 🤖 Discord Bot Setup

### 1. Create Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" → Name it
3. Go to "Bot" tab → "Add Bot"
4. Copy the **Bot Token**
5. Enable "Message Content Intent"

### 2. Get Channel IDs

1. Discord Settings → Advanced → Enable Developer Mode
2. Right-click channel → "Copy ID"
3. You need 3 channels: new-images, variations, upscaled

### 3. Configure

Add to `.env`:
```env
DISCORD_BOT_TOKEN=your_token
DISCORD_CHANNEL_NEW_IMAGES=123456789
DISCORD_CHANNEL_VARIATIONS=123456789
DISCORD_CHANNEL_UPSCALED=123456789
```

Set in `settings.json`:
```json
{ "discordEnabled": true }
```

### 4. Invite Bot

1. OAuth2 → URL Generator
2. Scopes: `bot`
3. Permissions: `Send Messages`, `Attach Files`, `Read Message History`
4. Copy URL → Open in browser → Select server

---

## 🛠️ Troubleshooting

### "Out of Memory" Error

1. Enable DRAM Extension in Settings
2. Use smaller resolution (512×512)
3. Reduce steps (20-30)
4. Close other GPU applications
5. Switch to CPU mode temporarily

### "CUDA not found"

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
```

### "No module named 'torch'"

Make sure you're using the venv Python:
```bash
# Windows
.\venv\Scripts\python.exe src/start.py

# Not just
python src/start.py
```

### Progress Bar Not Updating

- Refresh the page
- Check WebSocket connection in browser console
- Restart the server

### Discord Bot Not Working

- Verify token is correct
- Check channel IDs exist
- Ensure bot has permissions
- Check bot is in the server

---

## 📁 File Locations

| Path | Description |
|------|-------------|
| `outputs/` | Generated images |
| `Logs/` | Server logs |
| `static/data/prompts_history.json` | Prompt history |
| `settings.json` | Runtime settings |
| `.env` | Environment variables |

---

## 💬 Support

Open an issue on GitHub with:
- OS and Python version
- GPU model and VRAM
- Full error message
- Steps to reproduce
