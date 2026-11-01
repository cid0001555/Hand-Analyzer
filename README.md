# gesture-esp-rs

Hand tracking and gesture input daemon written in Rust.

## Setup

Download the required ONNX models and font assets before running:

Windows:
```powershell
.\scripts\setup_models.ps1
```

Linux:
```bash
chmod +x scripts/setup_models.sh
./scripts/setup_models.sh
```

On Linux, grant `/dev/uinput` access if running without root:
```bash
sudo cp etc/99-gesture-esp-uinput.rules /etc/udev/rules.d/99-gesture-esp-uinput.rules
sudo modprobe uinput
sudo usermod -aG input,video "$USER"
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Build & Run

```bash
cargo run --release
```

Optional flags:
- `--headless`: run without opening the HUD window
- `--dry-run`: disable OS mouse/keyboard input injection
- `--debug-kinematics`: show joint angles on the overlay
- `--record <NAME>`: record 200 pose samples to CSV and train custom MLP weights
- `--train-only`: retrain MLP weights from `data/recorded_poses.csv`
- `--config <PATH>`: path to configuration file (defaults to `config/mappings.toml`)
