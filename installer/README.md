# System Installers

OS-specific installation scripts for the Causal Graph Compiler.

## Structure

- `linux/install.sh` - Linux installer
- `windows/install.ps1` - Windows installer (PowerShell)
- `macos/install.sh` - macOS installer

## What the Installer Does

1. Checks for Docker and Docker Compose
2. Installs Docker if missing (with user permission)
3. Sets up application and data directories
4. Adds the `cgc` CLI to system PATH
5. Starts containerized services

## Usage

### Linux
```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/causal-graph-compiler/main/installer/linux/install.sh | bash
```

### Windows
```powershell
iwr -useb https://raw.githubusercontent.com/yourusername/causal-graph-compiler/main/installer/windows/install.ps1 | iex
```

### macOS
```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/causal-graph-compiler/main/installer/macos/install.sh | bash
```

Implements: Requirements 24, 25
