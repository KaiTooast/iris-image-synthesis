<div align="center">

# 🧠 L.O.O.M.

### Local Operator of Open Minds

<p>
  <strong>AI Image Generation System with Web UI and Discord Bot Integration</strong>
</p>

<p>
  <a href="#features"><img src="https://img.shields.io/badge/Stable_Diffusion-XL-blue?style=for-the-badge" alt="Stable Diffusion XL"></a>
  <a href="#features"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="#features"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="#features"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#contributing">Contributing</a>
</p>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

</div>

## ⚡ Quick Start

<div align="center">

<table>
<tr>
<td width="33%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/internet.png" width="64" alt="Web UI"/>

### 🌐 Web UI Only

```bash
python src/start.py web
```

<sub>Access at: <a href="http://localhost:8000">localhost:8000</a></sub>

</td>
<td width="33%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/bot.png" width="64" alt="Discord Bot"/>

### 🤖 Discord Bot Only

```bash
python src/start.py bot
```

<sub>Bot will connect to your server</sub>

</td>
<td width="33%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/rocket.png" width="64" alt="Both Services"/>

### 🚀 Both Services

```bash
python src/start.py all
```

<sub>Run everything at once</sub>

</td>
</tr>
</table>

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## ✨ Features

<div align="center">

<table>
<tr>
<td width="50%">

### 🎨 Generation Features
- **Stable Diffusion XL** - State-of-the-art image generation
- **Quality Presets** - Multiple generation settings
- **Real-ESRGAN Upscaling** - Enhance image resolution
- **📱 Mobile Wallpapers** - Optimized aspect ratios (720x1280, 1080x1920)

</td>
<td width="50%">

### 🖥️ Interface Features
- **🌐 Modern Web UI** - Clean, responsive interface
- **🤖 Discord Integration** - Generate images directly in Discord
- **📊 Real-time Progress** - Live generation tracking
- **🖼️ Image Gallery** - Browse with full metadata

</td>
</tr>
</table>

</div>

<details>
<summary><b>🔍 View Technical Specifications</b></summary>

<br/>

<table>
<tr>
<td><b>Model</b></td>
<td>Stable Diffusion XL with custom anime models</td>
</tr>
<tr>
<td><b>Backend</b></td>
<td>FastAPI with async support</td>
</tr>
<tr>
<td><b>Frontend</b></td>
<td>Modern HTML5/CSS3/JavaScript</td>
</tr>
<tr>
<td><b>Discord Bot</b></td>
<td>discord.py with slash commands</td>
</tr>
<tr>
<td><b>Image Processing</b></td>
<td>Real-ESRGAN for upscaling</td>
</tr>
<tr>
<td><b>GPU Support</b></td>
<td>NVIDIA CUDA, AMD ROCm, Apple MPS</td>
</tr>
</table>

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 📦 Installation

<div align="center">

### Quick Install

</div>

```bash
# 1️⃣ Clone the repository
git clone https://github.com/KaiTooast/Local-Operator-of-Open-Minds.git

# 2️⃣ Navigate to directory
cd Local-Operator-of-Open-Minds

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Configure (optional - for Discord bot)
mkdir -p static/config
echo "YOUR_BOT_TOKEN" > static/config/bot_token.txt
```

<details>
<summary><b>📋 GPU Requirements</b></summary>

<br/>

<table>
<tr>
<th>VRAM</th>
<th>Recommended Resolution</th>
<th>Features Available</th>
</tr>
<tr>
<td>4GB</td>
<td>512x512</td>
<td>Basic generation with DRAM extension</td>
</tr>
<tr>
<td>6GB</td>
<td>512x768, 720x1280</td>
<td>Standard generation + variations</td>
</tr>
<tr>
<td>8GB</td>
<td>1024x768</td>
<td>High-quality + upscaling</td>
</tr>
<tr>
<td>10GB+</td>
<td>1080x1920</td>
<td>Full mobile wallpapers + all features</td>
</tr>
</table>

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 📁 Project Structure

```
📦 Local-Operator-of-Open-Minds
┣ 📂 src/
┃ ┣ 📜 start.py              # 🚀 Universal starter script
┃ ┣ 📂 backend/              # 🐍 Python backend services
┃ ┃ ┣ 📜 web_server.py       # 🌐 FastAPI web server
┃ ┃ ┣ 📜 discord_bot.py      # 🤖 Discord bot
┃ ┃ ┗ 📜 logger.py           # 📝 Logging utilities
┃ ┗ 📂 frontend/             # 🎨 HTML frontend
┃   ┣ 📜 index.html          # 🏠 Main generator UI
┃   ┗ 📜 gallery.html        # 🖼️ Image gallery
┣ 📂 static/                 # 🎭 Static assets
┣ 📂 outputs/                # 🖼️ Generated images
┗ 📜 requirements.txt        # 📋 Python dependencies
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🎯 Usage Examples

<div align="center">

<table>
<tr>
<td width="50%">

### Web Interface

1. Start the web server
2. Open `localhost:8000`
3. Enter your prompt
4. Adjust settings (steps, guidance, size)
5. Click **Generate**
6. View results in gallery

</td>
<td width="50%">

### Discord Bot

1. Invite bot to your server
2. Use `/generate` command
3. Provide prompt and settings
4. Bot posts to configured channels
5. React to upscale or create variations

</td>
</tr>
</table>

</div>

<details>
<summary><b>🎨 Example Prompts</b></summary>

<br/>

```
✅ Good Prompts:
• "anime girl with long blue hair, detailed eyes, fantasy background, high quality"
• "cyberpunk cityscape at night, neon lights, futuristic, 4k"
• "cute cat sitting on a bookshelf, cozy library, warm lighting"

⚠️ Tips:
• Be specific and descriptive
• Include quality keywords (detailed, high quality, 4k)
• Describe style, lighting, and mood
• Use negative prompts to avoid unwanted elements
```

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🤝 Contributing

We welcome contributions! Please check out our [Contributing Guide](CONTRIBUTING.md) for guidelines.

<div align="center">

[![Contributors](https://img.shields.io/github/contributors/KaiTooast/Local-Operator-of-Open-Minds?style=for-the-badge)](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/KaiTooast/Local-Operator-of-Open-Minds?style=for-the-badge)](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/KaiTooast/Local-Operator-of-Open-Minds?style=for-the-badge)](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/pulls)

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🙏 Acknowledgments

<div align="center">

<table>
<tr>
<td align="center">
<img src="https://img.icons8.com/fluency/48/000000/artificial-intelligence.png" alt="AI"/>
<br/>
<b>Stable Diffusion</b>
<br/>
<sub>Core generation model</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/48/000000/discord-logo.png" alt="Discord"/>
<br/>
<b>discord.py</b>
<br/>
<sub>Bot framework</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/48/000000/code.png" alt="FastAPI"/>
<br/>
<b>FastAPI</b>
<br/>
<sub>Web framework</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/48/000000/community.png" alt="Community"/>
<br/>
<b>AI Community</b>
<br/>
<sub>Inspiration & support</sub>
</td>
</tr>
</table>

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

<div align="center">

### ⭐ Star us on GitHub — it motivates us a lot!

**Made with ❤️ for the AI Art Community**

[Report Bug](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues) · [Request Feature](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues) · [Join Discord](https://discord.gg/your-invite)

</div>
