# 🎨 I.R.I.S.
### Intelligent Rendering & Image Synthesis

**A modular, local-first AI image generation system — built to be forked, extended and customized.**

I.R.I.S. is an **open-source Stable Diffusion platform** designed as a **foundation**, not a locked product.  
Think of it as **Linux for AI image generation**:  
you get a working system, but you decide how far it goes.

> ⚠️ This project runs **entirely on your own hardware**.  
> No cloud, no accounts, no telemetry.

---

## ✨ Key Philosophy

- 🧠 **Local-first** — everything runs on your machine
- 🔓 **Open Source** — MIT licensed, no restrictions
- 🧩 **Modular** — replace UI, backend, models, pipelines
- 🧪 **Experimental-friendly** — built for tinkering
- 🚀 **Production-capable** — WebSockets, progress streaming, APIs

This repository provides a **fully functional reference implementation**, not a closed product.

---

## 🖼️ Features Overview

### Core Features
- Modern **Web UI** (Generate, Gallery, Settings)
- **Multiple AI models** (anime, realistic, pixel art, SDXL)
- **Text-to-Image** & **Image-to-Image**
- **Real-time progress** via WebSockets
- **Prompt & image history logging**
- **NSFW prompt filtering**
- **CPU & low-VRAM GPU support**

### Advanced Features
- **DRAM Extension** (use system RAM for low VRAM GPUs)
- **Custom resolutions** (256×256 → 4096×4096)
- **CFG scale fine control**
- **Real-ESRGAN upscaling** (2× / 4× / 8×)
- **Discord bot integration**
- **Gallery live updates**
- **Automatic VRAM safety adjustments**

---

## 🚀 Quick Start

### Requirements

```

Python 3.9 – 3.11
GPU recommended (4 GB VRAM minimum)
CUDA 11.8+ (optional, CPU mode supported)

````

### Installation

```bash
git clone https://github.com/KaiTooast/iris-image-synthesis.git
cd iris-image-synthesis

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
````

### Run

```bash
# Web UI only
python src/start.py web

# Discord bot only
python src/start.py bot

# Everything
python src/start.py all
```

🌐 Open: **[http://localhost:8000](http://localhost:8000)**

---

## 🧩 Project Structure

```
src/
├── api/            # FastAPI + WebSocket backend
├── core/           # Model loading & generation logic
├── services/       # Discord, upscaling, extensions
├── utils/          # Logging, file handling
├── frontend/       # HTML UI (Generate, Gallery, Settings)
└── start.py        # Unified entry point
```

Static data, outputs and logs are **explicitly separated** for easy modification.

---

## ⚙️ Configuration (.env)

```env
HOST=0.0.0.0
PORT=8000

DEFAULT_MODEL=anime_kawai

DRAM_EXTENSION_ENABLED=false
VRAM_THRESHOLD_GB=6
MAX_DRAM_GB=16
```

Discord integration is **optional** and fully isolated.

---

## 🎯 Designed for Modification

You are encouraged to:

* Replace the frontend entirely
* Add your own models or pipelines
* Build a token / subscription system
* Deploy in a datacenter
* Run on NVIDIA, AMD, Intel (experimental)
* Fork this into a commercial or private project

This repository **intentionally does not enforce a business model**.

---

## 🖥️ Hardware Performance (Reference)

| Tier        | GPU      | VRAM  | Notes              |
| ----------- | -------- | ----- | ------------------ |
| Minimum     | GTX 1650 | 4 GB  | Tested & supported |
| Recommended | RTX 3060 | 12 GB | Smooth experience  |
| High-End    | RTX 4090 | 24 GB | Near real-time     |

> I.R.I.S. was **tested on a GTX 1650**, proving the system works even on low-end hardware.

---

## 🔌 API & WebSocket Support

* REST API for generation, gallery, system info
* WebSocket streaming for:

  * Generation progress
  * Gallery updates
  * Multi-page synchronization

Perfect for **custom frontends** or external clients.

---

## 🛡️ Safety & Filters

* Prompt-based NSFW filtering
* Explicit content blocking
* Easily extendable keyword system
* Can be disabled per request

---

## 📜 License

**MIT License**
Use it, fork it, sell it, modify it — just keep the license.

---

## 🤝 Contributing

Contributions are welcome — from small fixes to major rewrites.

See **CONTRIBUTING.md** for:

* Code style
* Architecture notes
* Model integration rules

---

## 💬 Final Note

I.R.I.S. is not meant to compete with cloud AI platforms.
It exists to **give people control back** over AI image generation.

If you want:

* freedom instead of subscriptions
* experimentation instead of lock-in
* ownership instead of APIs

Then this project is for you.

