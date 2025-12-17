<div align="center">

# 🤝 Contributing to L.O.O.M.

### Local Operator of Open Minds

<p>
<strong>Thank you for considering contributing! It's people like you that make this project better.</strong>
</p>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

</div>

## 📑 Table of Contents

<details open>
<summary><b>Quick Navigation</b></summary>

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Development Setup](#-development-setup)
- [Pull Request Process](#-pull-request-process)
- [Coding Standards](#-coding-standards)
- [Testing Guidelines](#-testing-guidelines)
- [Configuration & Discord Setup](#-configuration--discord-setup)
- [Getting Help](#-getting-help)

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 📜 Code of Conduct

<div align="center">

> **TL;DR**: Be respectful, inclusive, and professional.

</div>

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

<table>
<tr>
<td>✅ <b>Do</b></td>
<td>❌ <b>Don't</b></td>
</tr>
<tr>
<td>• Be welcoming and friendly<br/>• Respect differing viewpoints<br/>• Accept constructive criticism<br/>• Focus on what's best for the community</td>
<td>• Use inappropriate language<br/>• Troll or insult others<br/>• Harass or discriminate<br/>• Share private information</td>
</tr>
</table>

> **Note:** L.O.O.M. is dedicated to anime art generation and has built-in safety filters for NSFW content.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🎯 How Can I Contribute?

<div align="center">

<table>
<tr>
<td width="33%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/bug.png" width="64" alt="Bug Reports"/>

### 🐛 Bug Reports

Find and report issues to help us improve

</td>
<td width="33%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/idea.png" width="64" alt="Feature Requests"/>

### 💡 Feature Requests

Suggest new ideas and enhancements

</td>
<td width="33%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/code.png" width="64" alt="Code Contributions"/>

### 👨‍💻 Code Contributions

Submit pull requests with improvements

</td>
</tr>
</table>

</div>

### 🐛 Reporting Bugs

<details>
<summary><b>Before submitting a bug report</b></summary>

<br/>

- ✅ Check [existing issues](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues)
- ✅ Test with the latest version
- ✅ Gather environment details (OS, GPU, Python version, CUDA version)
- ✅ Try to reproduce the bug consistently

</details>

<details open>
<summary><b>Bug Report Template</b></summary>

```markdown
**🐛 Describe the bug**
A clear and concise description of what the bug is.

**📝 To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**✅ Expected behavior**
A clear and concise description of what you expected to happen.

**📸 Screenshots**
If applicable, add screenshots to help explain your problem.

**💻 Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- GPU: [e.g., RTX 3060, GTX 1660 Ti]
- Python Version: [e.g., 3.11.5]
- CUDA Version: [e.g., 12.1]
- VRAM: [e.g., 6GB]

**📋 Additional context**
Add any other context about the problem here.
```

</details>

### 💡 Suggesting Features

<details open>
<summary><b>Feature Request Template</b></summary>

```markdown
**❓ Is your feature request related to a problem?**
A clear and concise description of what the problem is.
Example: "I'm always frustrated when..."

**💡 Describe the solution you'd like**
A clear and concise description of what you want to happen.

**🔄 Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**🎨 Related to L.O.O.M.**
How does this feature fit with anime art generation and L.O.O.M.'s goals?

**📎 Additional context**
Add any other context, mockups, or screenshots about the feature request here.
```

</details>

### 💻 Code Contributions We Accept

<table>
<tr>
<td width="50%">

#### ✅ We Accept

- 🐛 Bug fixes
- ✨ New features (discuss first)
- ⚡ Performance improvements
- 🎨 UI/UX enhancements
- 🤖 Better Discord integration
- 🧠 Anime model improvements
- 💾 Memory optimization
- 📚 Documentation improvements

</td>
<td width="50%">

#### ❌ We Don't Accept

- 🔞 NSFW content generation features
- 🛡️ Removal of safety filters
- ⛏️ Cryptocurrency mining
- 🦠 Malicious code
- ⚠️ Code without tests (for core features)
- 📝 Poor documentation

</td>
</tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🛠️ Development Setup

### Step 1️⃣: Fork & Clone

```bash
# Fork the repository on GitHub, then clone your fork:
git clone https://github.com/YOUR_USERNAME/Local-Operator-of-Open-Minds.git
cd Local-Operator-of-Open-Minds

# Add upstream remote to sync with main repo:
git remote add upstream https://github.com/KaiTooast/Local-Operator-of-Open-Minds.git
```

### Step 2️⃣: Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/bug-description
```

<details>
<summary><b>🏷️ Branch Naming Convention</b></summary>

<br/>

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/model-selector` |
| `fix/` | Bug fixes | `fix/discord-connection` |
| `docs/` | Documentation | `docs/update-setup-guide` |
| `refactor/` | Code refactoring | `refactor/optimize-memory` |
| `test/` | Adding tests | `test/add-safety-tests` |
| `perf/` | Performance | `perf/gpu-acceleration` |

</details>

### Step 3️⃣: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

# Install development dependencies (if available)
pip install -r requirements-dev.txt
```

### Step 4️⃣: Configure for Development

```bash
# Create config directories
mkdir -p static/config
mkdir -p static/data

# Add your Discord bot token
echo "YOUR_BOT_TOKEN_HERE" > static/config/bot_token.txt

# Add your Discord user ID
echo "YOUR_DISCORD_ID_HERE" > static/config/bot_owner_id.txt

# Add bot's user ID
echo "BOT_USER_ID_HERE" > static/config/bot_id.txt

# Configure Discord channels (see Configuration section below)
cat > static/config/channel_ids.txt << EOF
new=CHANNEL_ID_HERE
variations=CHANNEL_ID_HERE
upscaled=CHANNEL_ID_HERE
EOF
```

> **💡 Tip:** See [Configuration & Discord Setup](#-configuration--discord-setup) for detailed instructions.

### Step 5️⃣: Test Your Setup

```bash
# Test web UI only
python src/start.py web
# Then visit http://localhost:8000

# Test bot only
python src/start.py bot

# Test everything
python src/start.py all
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🔧 Configuration & Discord Setup

<div align="center">

### Discord Bot Setup Guide

</div>

#### Step 1: Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **"New Application"**
3. Give it a name (e.g., "L.O.O.M. Bot")
4. Navigate to **"Bot"** section
5. Click **"Add Bot"**

#### Step 2: Enable Required Intents

<table>
<tr>
<th>Intent</th>
<th>Required For</th>
<th>Status</th>
</tr>
<tr>
<td>Message Content Intent</td>
<td>Reading message content</td>
<td>✅ Required</td>
</tr>
<tr>
<td>Server Members Intent</td>
<td>Accessing member information</td>
<td>✅ Required</td>
</tr>
<tr>
<td>Presence Intent</td>
<td>User status (optional)</td>
<td>⚠️ Optional</td>
</tr>
</table>

#### Step 3: Get Required IDs

<details>
<summary><b>How to Get Discord IDs</b></summary>

<br/>

1. **Enable Developer Mode:**
   - User Settings → App Settings → Advanced
   - Toggle "Developer Mode" ON

2. **Get Bot User ID:**
   - Copy from Application ID in Developer Portal

3. **Get Your User ID:**
   - Right-click your username → Copy User ID

4. **Get Channel IDs:**
   - Right-click channel → Copy Channel ID

</details>

#### Step 4: Create Test Server & Channels

```
📁 Your Test Server
├── 📢 #general
├── 🎨 #generated-images     ← For new generations
├── 🔄 #variations           ← For image variations
└── ⬆️ #upscaled             ← For upscaled images
```

#### Step 5: Configure L.O.O.M.

<details open>
<summary><b>Configuration File Format</b></summary>

```ini
# static/config/channel_ids.txt

# Channel for new generated images
new=123456789012345678

# Channel for variations of existing images
variations=234567890123456789

# Channel for upscaled images
upscaled=345678901234567890
```

</details>

<div align="center">

### 📂 Configuration Files Overview

</div>

```
static/config/
├── bot_token.txt        # Your Discord bot token
├── bot_owner_id.txt     # Your Discord user ID
├── bot_id.txt           # Bot's user ID
└── channel_ids.txt      # Discord channel IDs (customizable)
```

> ⚠️ **Security:** Never commit config files to Git! They're in `.gitignore` by default.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🔀 Pull Request Process

### Step 1️⃣: Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add anime model selector to UI"
```

<details>
<summary><b>📝 Commit Message Convention</b></summary>

<br/>

**Format:**
```
<type>(<scope>): <short summary>

<optional detailed description>

<optional footer>
```

**Types:**
| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat: add custom negative prompts` |
| `fix` | Bug fix | `fix: resolve Discord channel validation` |
| `docs` | Documentation | `docs: update GPU compatibility guide` |
| `style` | Code formatting | `style: format code with black` |
| `refactor` | Code restructuring | `refactor: optimize NSFW filter` |
| `test` | Adding tests | `test: add unit tests for safety filter` |
| `chore` | Maintenance | `chore: update dependencies` |
| `perf` | Performance | `perf: optimize memory usage` |

**Examples:**
```bash
✅ feat(ui): add model selector dropdown with preview
✅ fix(discord): resolve channel ID validation issue
✅ docs(readme): add GPU compatibility matrix
✅ refactor(safety): optimize NSFW filter performance
```

</details>

### Step 2️⃣: Push to Your Fork

```bash
# Push your branch to your fork
git push origin feature/your-feature-name
```

### Step 3️⃣: Create Pull Request

<details open>
<summary><b>📋 Pull Request Template</b></summary>

```markdown
## 📝 Description
Brief explanation of what this PR does and why it's needed.

## 🔗 Related Issue
Fixes #123 (if applicable)
Closes #456 (if applicable)

## 🎯 Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update

## 🧪 How Has This Been Tested?
Describe the tests you ran to verify your changes:
- [ ] Tested locally on Windows/Linux/Mac
- [ ] Tested with GPU (specify model)
- [ ] Tested Discord integration
- [ ] Added unit tests
- [ ] Tested with different VRAM configurations

## 📸 Screenshots (if applicable)
Add before/after screenshots for UI changes.

## ✅ Checklist
- [ ] Code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] NSFW safety filters are still active and functioning
- [ ] Discord channel configuration remains flexible
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## 📋 Additional Notes
Any additional information, concerns, or context.
```

</details>

### Step 4️⃣: Code Review & Iteration

<table>
<tr>
<td>

**During Review:**
- ✅ Respond to feedback promptly
- ✅ Make requested changes
- ✅ Push updates to the same branch
- ✅ Be open to suggestions
- ✅ Ask questions if unclear

</td>
<td>

**After Approval:**
- 🎉 Maintainers will merge your PR
- 🌟 You'll be added to contributors
- 📣 Changes will be in the next release
- 💚 Thank you for contributing!

</td>
</tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 📏 Coding Standards

### Python Style Guide

<div align="center">

**Follow [PEP 8](https://peps.python.org/pep-0008/) with project-specific guidelines**

</div>

#### ✅ Good Code Example

```python
from typing import Optional

def apply_nsfw_filter(prompt: str, negative_prompt: str) -> str:
    """
    Apply NSFW safety filter to prompt and negative prompt.
    
    Always appends safety keywords to negative prompt to prevent
    unsafe content generation. This is a critical security feature.
    
    Args:
        prompt: Main generation prompt from user
        negative_prompt: Existing negative prompt (can be empty)
    
    Returns:
        str: Filtered negative prompt with safety keywords appended
        
    Example:
        >>> apply_nsfw_filter("anime girl", "low quality")
        "low quality, nsfw, nude, naked, explicit, sexual"
    """
    NSFW_KEYWORDS = "nsfw, nude, naked, explicit, sexual, adult content"
    
    if negative_prompt:
        return f"{negative_prompt}, {NSFW_KEYWORDS}"
    return NSFW_KEYWORDS
```

#### ❌ Bad Code Example

```python
# Bad - No types, no documentation, unclear logic
def filter_prompt(p, n):
    return f"{n}, nsfw, nude".strip() if n else "nsfw, nude"
```

### Key Coding Rules

<table>
<tr>
<td width="50%">

#### ✅ Required

- ✔️ Use type hints for all functions
- ✔️ Write docstrings (Google style)
- ✔️ Meaningful variable names
- ✔️ Max line length: 100 characters
- ✔️ Use f-strings for formatting
- ✔️ Import order: stdlib → third-party → local
- ✔️ Always preserve NSFW safety filters

</td>
<td width="50%">

#### ❌ Avoid

- ❌ Single-letter variables (except i, j, k in loops)
- ❌ Magic numbers without constants
- ❌ Nested functions >3 levels deep
- ❌ Functions >50 lines (split them)
- ❌ Global mutable state
- ❌ Hardcoded configuration values
- ❌ Removing safety features

</td>
</tr>
</table>

### File Organization

```python
# ✅ Good import order

# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Third-party packages
import torch
import discord
from fastapi import FastAPI
from diffusers import StableDiffusionPipeline

# Local imports
from src.backend.logger import setup_logger
from src.backend.config import load_config
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_safety.py

# Run with coverage report
pytest --cov=src tests/

# Run with verbose output
pytest -v

# Run specific test function
pytest tests/test_safety.py::test_nsfw_filter
```

### Unit Testing Example

<details>
<summary><b>Example: Safety Filter Tests</b></summary>

```python
# tests/test_safety.py
import pytest
from src.backend.web_server import apply_nsfw_filter, NSFW_KEYWORDS

class TestNSFWFilter:
    """Test suite for NSFW safety filter."""
    
    def test_nsfw_filter_exists(self):
        """Verify NSFW filter is defined and contains required keywords."""
        assert NSFW_KEYWORDS is not None
        assert "nsfw" in NSFW_KEYWORDS.lower()
        assert "nude" in NSFW_KEYWORDS.lower()
        assert "explicit" in NSFW_KEYWORDS.lower()
    
    def test_nsfw_filter_applied(self):
        """Test that NSFW filter is applied to negative prompt."""
        result = apply_nsfw_filter("anime girl", "low quality")
        assert "nsfw" in result.lower()
        assert "low quality" in result
    
    def test_nsfw_filter_empty_negative(self):
        """Test NSFW filter with empty negative prompt."""
        result = apply_nsfw_filter("anime girl", "")
        assert "nsfw" in result.lower()
        assert len(result) > 0
    
    @pytest.mark.parametrize("prompt,negative", [
        ("cute cat", "blurry"),
        ("landscape", ""),
        ("portrait", "low quality, bad anatomy"),
    ])
    def test_nsfw_filter_various_inputs(self, prompt, negative):
        """Test NSFW filter with various input combinations."""
        result = apply_nsfw_filter(prompt, negative)
        assert "nsfw" in result.lower()
```

</details>

<details>
<summary><b>Example: Discord Integration Tests</b></summary>

```python
# tests/test_discord.py
import pytest
from pathlib import Path
from src.backend.discord_bot import load_channel_ids

class TestDiscordConfiguration:
    """Test suite for Discord bot configuration."""
    
    def test_channel_ids_loaded(self, tmp_path):
        """Test that channel IDs are loaded from config file."""
        # Create temporary config file
        config_file = tmp_path / "channel_ids.txt"
        config_file.write_text("new=123\nvariations=456\nupscaled=789")
        
        # Load configuration
        channels = load_channel_ids(config_file)
        
        # Verify
        assert channels["new"] == "123"
        assert channels["variations"] == "456"
        assert channels["upscaled"] == "789"
    
    def test_no_hardcoded_channel_ids(self):
        """Ensure no hardcoded channel IDs exist in bot code."""
        # Read bot source code
        bot_code = Path("src/backend/discord_bot.py").read_text()
        
        # Check for hardcoded channel IDs (18-19 digit numbers)
        import re
        hardcoded_ids = re.findall(r'\b\d{18,19}\b', bot_code)
        
        # Allow only in comments or config loading
        assert len(hardcoded_ids) == 0, f"Found hardcoded IDs: {hardcoded_ids}"
```

</details>

### Test Coverage Goals

<table>
<tr>
<th>Component</th>
<th>Target Coverage</th>
<th>Priority</th>
</tr>
<tr>
<td>Safety Filters</td>
<td>>95%</td>
<td>🔴 Critical</td>
</tr>
<tr>
<td>Core Generation</td>
<td>>80%</td>
<td>🟠 High</td>
</tr>
<tr>
<td>Discord Bot</td>
<td>>70%</td>
<td>🟡 Medium</td>
</tr>
<tr>
<td>Web UI Backend</td>
<td>>70%</td>
<td>🟡 Medium</td>
</tr>
<tr>
<td>Utilities</td>
<td>>60%</td>
<td>🟢 Low</td>
</tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🖥️ GPU Compatibility

<div align="center">

### Supported Hardware

</div>

<table>
<tr>
<th>Manufacturer</th>
<th>Models</th>
<th>Compute</th>
<th>Status</th>
</tr>
<tr>
<td rowspan="3"><b>NVIDIA</b></td>
<td>RTX 40/30/20 Series</td>
<td>Float16, Tensor Cores</td>
<td>✅ Full Support</td>
</tr>
<tr>
<td>GTX 16/10 Series</td>
<td>Float32</td>
<td>✅ Supported</td>
</tr>
<tr>
<td>Datacenter (A100, V100, T4)</td>
<td>Mixed Precision</td>
<td>✅ Optimized</td>
</tr>
<tr>
<td><b>AMD</b></td>
<td>RX 6000/7000 Series</td>
<td>ROCm</td>
<td>⚠️ Limited</td>
</tr>
<tr>
<td><b>Apple</b></td>
<td>M1/M2/M3</td>
<td>MPS</td>
<td>✅ Supported</td>
</tr>
<tr>
<td><b>Intel</b></td>
<td>Arc A-Series</td>
<td>XPU</td>
<td>⚠️ Experimental</td>
</tr>
<tr>
<td><b>CPU</b></td>
<td>Any</td>
<td>CPU Mode</td>
<td>⚠️ Very Slow</td>
</tr>
</table>

### VRAM Requirements

<table>
<tr>
<th>VRAM</th>
<th>Resolution</th>
<th>Quality</th>
<th>Recommendations</th>
</tr>
<tr>
<td><b>4GB</b></td>
<td>512×512</td>
<td>Basic</td>
<td>Enable DRAM extension, lower batch size</td>
</tr>
<tr>
<td><b>6GB</b></td>
<td>512×768<br/>720×1280</td>
<td>Standard</td>
<td>Most features available, some limitations</td>
</tr>
<tr>
<td><b>8GB</b></td>
<td>768×1024</td>
<td>High</td>
<td>All features, comfortable usage</td>
</tr>
<tr>
<td><b>10GB+</b></td>
<td>1080×1920</td>
<td>Ultra</td>
<td>Full resolution mobile wallpapers</td>
</tr>
<tr>
<td><b>12GB+</b></td>
<td>1080×1920+</td>
<td>Maximum</td>
<td>Batch processing, multiple models</td>
</tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## ❓ Getting Help

<div align="center">

### We're Here to Help!

</div>

<table>
<tr>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/github.png" width="48" alt="Issues"/>

### 🐛 Issues
[Report bugs](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues)

</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/chat.png" width="48" alt="Discussions"/>

### 💬 Discussions
[Ask questions](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/discussions)

</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/discord.png" width="48" alt="Discord"/>

### 💬 Discord
[Join community](#)

</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/book.png" width="48" alt="Docs"/>

### 📚 Docs
[Read guides](docs/)

</td>
</tr>
</table>

### 📖 Additional Resources

- 📘 [Setup Guide](docs/SETUP.md) - Detailed installation instructions
- 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- 🤖 [Discord.py Docs](https://discordpy.readthedocs.io/) - Bot framework documentation
- ⚡ [FastAPI Docs](https://fastapi.tiangolo.com/) - Web framework documentation
- 🎨 [Diffusers Docs](https://huggingface.co/docs/diffusers/) - AI model library

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 🏆 Recognition

<div align="center">

### Contributors are Recognized In:

</div>

<table>
<tr>
<td width="33%" align="center">

**📝 README.md**

Contributors section with avatars

</td>
<td width="33%" align="center">

**📦 Release Notes**

Mentioned in version releases

</td>
<td width="33%" align="center">

**📊 GitHub Profile**

Contributors graph and stats

</td>
</tr>
</table>

<div align="center">

### 🌟 Top Contributors

[![Contributors](https://contrib.rocks/image?repo=KaiTooast/Local-Operator-of-Open-Minds)](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/graphs/contributors)

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

## 📄 License

<div align="center">

By contributing, you agree that your contributions will be licensed under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

<div align="center">

### 🎉 Thank You!

**Thank you for contributing to L.O.O.M.!** Every contribution helps improve anime art generation for everyone. Let's build something amazing together! 🎨

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="separator" width="100%"/>

**Made with ❤️ by the L.O.O.M. Community**

[⭐ Star this repo](https://github.com/KaiTooast/Local-Operator-of-Open-Minds) · [🐛 Report Bug](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues) · [💡 Request Feature](https://github.com/KaiTooast/Local-Operator-of-Open-Minds/issues)

</div>
