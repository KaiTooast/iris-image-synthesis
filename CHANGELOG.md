# Changelog

All notable changes to I.R.I.S. will be documented in this file.

---

## [1.2.0] - 2026-01-14

### 🎉 Major Update - Admin Panel & System Improvements

---

### ✨ New Features

#### Admin Panel
- **Protected Admin Panel** at `/app/admin` with password authentication
- **System Overview** — Real-time GPU, CPU, RAM stats
- **Server Logs** — View recent server activity
- **Quick Actions** — Clear CUDA cache, reload system info, garbage collection
- **Danger Zone** — Server restart, force clear all caches
- **Model Unload** — Free VRAM by unloading the current model

#### Server Mode System
- New `--mode` argument replaces `--no-html`
- **api** — API only (for external frontends)
- **html** — API + HTML frontend (default)
- **react** — API + React production build
- **full** — API + HTML + React

#### Image Variations
- New `/api/variation` endpoint for img2img variations
- Adjustable strength, steps, CFG scale
- Variations saved with `var_` prefix

#### Improved Restart System
- **CTRL+R** hotkey for instant server restart (Windows)
- **Admin Panel restart** via signal file
- Reliable process termination with taskkill

#### Enhanced Download Filenames
- Custom filenames include image type detection
- Prefixes: `normal`, `upscaled`, `variation`
- Format: `IRIS_{type}_{seed}_{date}.png`

---

### 🛠️ Improvements

#### Dashboard (formerly Settings)
- Renamed from "Settings" to "Dashboard"
- Route changed from `/settings` to `/dashboard`
- **Autosave** — All settings save automatically (500ms debounce)
- **Improved Toggle** — Better sizing and styling
- Reduced polling frequency (30s instead of 2s)

#### Discord Bot
- **Force Shutdown** — Instant kill with taskkill on Windows
- Improved process management

#### GPU Info Endpoint
- New `/api/gpu-info` with detailed stats
- GPU temperature, utilization, power draw
- CPU cores, frequency, usage
- RAM total, used, free, percentage

#### Model Performance Comments
- Detailed VRAM requirements in pipeline.py
- Quality/speed ratings for each model
- Performance tiers: S-TIER, A-TIER, B-TIER

#### UI Improvements
- All emojis replaced with SVG icons
- Consistent icon styling across all pages
- HomePage uses Sidebar component

---

### � Bug Fixes

- Fixed `seed=None` causing `RuntimeError` in generation
- Fixed Discord toggle autosave conflict
- Fixed CTRL+R restart not working
- Fixed Dashboard values not loading
- Fixed navigation still showing "Settings" instead of "Dashboard"

---

### 📦 API Changes

#### New Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/variation` | POST | Create image variation |
| `/api/gpu-info` | GET | Detailed GPU/CPU/RAM stats |
| `/api/admin/logs` | GET | Server logs |
| `/api/admin/clear-cache` | POST | Clear CUDA cache |
| `/api/admin/restart` | POST | Restart server |
| `/api/admin/stats` | GET | Admin statistics |
| `/api/admin/gc` | POST | Run garbage collection |
| `/api/admin/unload-model` | POST | Unload current model |

#### Updated Endpoints
- `/api/health` — Now includes uptime and generation stats

---

## [1.0.0] - 2026-01-13

### 🎉 First Stable Release

This is the first stable release of I.R.I.S., featuring a complete local AI image generation platform with dual frontend support.

---

### ✨ Features

#### Dual Frontend System
- **React Frontend** — Modern, responsive UI built with React 18, Vite, and Tailwind CSS
- **HTML Frontend** — Classic lightweight UI for maximum compatibility

#### Multi-GPU Support
- NVIDIA CUDA, AMD ROCm, Intel Arc XPU, Apple Silicon MPS, CPU Fallback

#### Advanced Upscaling
- Real-ESRGAN, Anime v3, Tile Mode, Lanczos (2x and 4x)

#### Discord Integration
- Auto-post images, Rich Presence, separate channels

#### NSFW Filter
- Three strength levels, category-based detection

#### DRAM Extension
- System RAM fallback for low-VRAM GPUs (4GB+)

---

### 📦 Supported Models

- Anime Kawai, Stable Diffusion 2.1/3.5, FLUX.1 Schnell
- OpenJourney, Pixel Art, Pony Diffusion v6
- Anything v5, Animagine XL 3.1, AbyssOrangeMix3, Counterfeit v3

---

### 🙏 Acknowledgments

Built with Stable Diffusion, Diffusers, Real-ESRGAN, FastAPI, React, Tailwind CSS
