# 🌌 I.R.I.S.
### Intelligent Rendering & Image Synthesis

**A modular, local-first AI image generation engine — built to be forked, extended, and owned.**

I.R.I.S. is an **open-source AI image generation platform** designed as a **foundation**, not a locked product.  
Think of it as **Linux for AI image generation**:

> You get a fully working system —  
> but *you* decide how it evolves.

⚠️ **Runs entirely on your own hardware**  
No cloud. No accounts. No telemetry. No vendor lock-in.

---

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![WebSockets](https://img.shields.io/badge/WebSockets-realtime-purple)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-v1.2.0-brightgreen)

---

## ✨ Core Philosophy

- 🧠 **Local-first** — everything runs on your machine
- 🔓 **Open Source** — modify, fork, redistribute
- 🧩 **Modular architecture** — UI, backend, models are replaceable
- 🧪 **Experiment-friendly** — designed for tinkering & research
- 🚀 **Production-capable** — APIs, WebSockets, scaling-ready

This repository provides a **fully functional reference implementation**, not a closed product.

---

## 🖼️ Feature Overview

### Core Features
- **Dual Frontend** — Classic HTML UI + Modern React UI
- **Multiple AI models** (anime, realistic, pixel art, SDXL)
- **Text-to-Image** generation with real-time progress
- **WebSocket streaming** for live updates
- **Persistent prompt history** (server-side)
- **NSFW prompt filtering** (configurable, can be disabled)
- **Multi-GPU support** (NVIDIA CUDA, AMD ROCm, Intel Arc XPU, Apple MPS, CPU)

### Advanced Features
- **DRAM Extension** — System RAM fallback for low VRAM GPUs (4GB+)
- **Multiple Upscalers** — Real-ESRGAN, Anime v3, Tile Mode, Lanczos
- **Custom resolutions** (256×256 → 2048×2048)
- **Hardware monitoring** — CPU, RAM, GPU power draw
- **Device switching** — Switch between GPU/CPU at runtime
- **Discord bot integration** — Auto-post generated images
- **Discord Rich Presence** — Show generation status

---

## 🚀 Quick Start

### Requirements
- Python 3.9 – 3.11
- GPU recommended (4 GB VRAM minimum)
- CUDA 11.8+ / ROCm 5.6+ / oneAPI (optional, CPU mode supported)

### Installation
```bash
git clone https://github.com/KaiTooast/iris.git
cd iris

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt

# Optional: Copy environment template
cp .env.example .env
```

### Run
```bash
# Windows (use venv python directly)
.\venv\Scripts\python.exe src/start.py

# Linux / macOS
python src/start.py

# Server Modes
python src/start.py --mode api      # API only
python src/start.py --mode html     # API + HTML (default)
python src/start.py --mode react    # API + React build
python src/start.py --mode full     # All frontends

# Without Discord bot
python src/start.py --no-bot
```

🌐 **HTML Frontend:** [http://localhost:8000](http://localhost:8000)  
🌐 **React Frontend:** [http://localhost:8000/app](http://localhost:8000/app) (with `--mode react`)  
🔐 **Admin Panel:** [http://localhost:8000/app/admin](http://localhost:8000/app/admin)

---

## 🧩 Project Structure

```
iris/
├── src/                    # Backend & core logic
│   ├── api/                # FastAPI server & routes
│   │   ├── server.py       # Main server (generation, upscaling, settings)
│   │   ├── middleware/     # Rate limiting
│   │   ├── routes/         # API endpoints (system, devices)
│   │   └── services/       # NSFW filter, pipeline, history, queue
│   ├── core/               # Model loading & generation
│   ├── services/           # Discord bot
│   ├── utils/              # Logging, file management
│   └── start.py            # Entry point
│
├── frontend/               # Classic HTML Web UI
│   ├── index.html          # Landing page
│   ├── generate.html       # Generation UI
│   ├── gallery.html        # Image gallery
│   └── settings.html       # Settings page
│
├── frontend-react/         # Modern React Web UI
│   ├── src/
│   │   ├── pages/          # HomePage, GeneratePage, GalleryPage, SettingsPage
│   │   ├── components/     # Reusable components
│   │   ├── store/          # Zustand state management
│   │   └── lib/            # API utilities
│   ├── package.json
│   └── vite.config.js
│
├── static/                 # Static assets & runtime data
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript
│   ├── config/             # Bot config files
│   └── data/               # History (prompts_history.json)
│
├── outputs/                # Generated images
├── Logs/                   # Runtime logs
├── docs/                   # Documentation
│
├── settings.json           # Runtime settings
└── requirements.txt        # Python dependencies
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

| Setting | Description |
|---------|-------------|
| `dramEnabled` | Use system RAM when VRAM is low |
| `vramThreshold` | VRAM threshold (GB) to enable DRAM Extension |
| `maxDram` | Maximum system RAM to use (GB) |
| `nsfwEnabled` | Enable/disable NSFW prompt filter |
| `nsfwStrength` | 1=Minimal, 2=Standard, 3=Strict |
| `discordEnabled` | Auto-start Discord bot |

### .env (optional)
```env
HOST=0.0.0.0
PORT=8000
DEFAULT_MODEL=anime_kawai

# Discord Bot (optional)
DISCORD_BOT_TOKEN=your_token
DISCORD_CHANNEL_NEW_IMAGES=channel_id
DISCORD_CHANNEL_VARIATIONS=channel_id
DISCORD_CHANNEL_UPSCALED=channel_id
```

---

## 🖥️ Hardware Reference

| Tier | GPU | VRAM | Notes |
|------|-----|------|-------|
| **Minimum** | NVIDIA GTX 1650 | 4 GB | The birthplace. Small models, DRAM Extension recommended. |
| **Sweet Spot** | **Intel Arc B580** | 12 GB | **Best value for money.** |
| **Advanced** | NVIDIA RTX 4070 Super | 12 GB | Faster inference, still VRAM-limited. |
| **Professional** | NVIDIA RTX 3090 Ti / 4090 | 24 GB | No-compromise local AI & SDXL. |
| **God Tier** | **NVIDIA RTX 5090** | 32 GB | Industrial scale. (Overkill for most) |

> 💡 **Developer Note:** I.R.I.S. was **developed and tested on a GTX 1650**, proving functionality on low-end hardware. We optimize for best hardware per dollar, not expensive branding.

---

## 🔌 API & WebSocket Support

- REST API for generation, gallery, system info
- WebSocket streams for:
  - Generation progress
  - Gallery updates
  - Multi-page synchronization

Perfect for **custom frontends**, automation, or external clients.

---

## 🛡️ Safety

- Prompt-based NSFW filtering
- Three strength levels (Minimal, Standard, Strict)
- Category-based detection
- Easily extendable or disableable

---

## 🧠 Designed for Modification

You are explicitly encouraged to:

* Replace the frontend entirely
* Add your own models or pipelines
* Build a token or subscription system
* Deploy in a private or public datacenter
* Run on NVIDIA, AMD, or Intel GPUs (experimental)
* Fork this into a commercial or closed product

**I.R.I.S. does not enforce a business model.**

---

## � License

**Creative Commons Attribution 4.0 (CC BY 4.0)**

You may use, modify, redistribute, and commercialize this project —
**attribution is required.**

See `LICENSE` for details.

---

## 🤝 Contributing

Contributions are welcome — from small fixes to major architectural changes.

Please read **CONTRIBUTING.md** before submitting a pull request.

---

## 🌍 Final Note

I.R.I.S. is not built to compete with cloud AI platforms.

It exists to **give control back** to developers and creators.

If you value:
- ownership over subscriptions
- experimentation over lock-in
- transparency over black boxes

then this project is for you.
