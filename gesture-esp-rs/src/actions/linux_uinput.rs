#![cfg(target_os = "linux")]

use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use evdev::uinput::{VirtualDevice, VirtualDeviceBuilder};
use evdev::{
    AbsInfo, AbsoluteAxisType, AttributeSet, EventType, InputEvent, Key, RelativeAxisType,
    UinputAbsSetup,
};
use parking_lot::Mutex;

use crate::actions::OsController;
use crate::fsm::{MediaCmd, MouseBtn, VirtualKey};

const ABS_MAX: i32 = 65535;

pub struct LinuxUinputController {
    device: Mutex<VirtualDevice>,
    use_pactl_fallback: bool,
}

impl LinuxUinputController {
    pub fn new() -> Result<Self> {
        let mut keys = AttributeSet::<Key>::new();

        keys.insert(Key::BTN_LEFT);
        keys.insert(Key::BTN_RIGHT);
        keys.insert(Key::BTN_MIDDLE);

        for k in [
            Key::KEY_LEFTMETA,
            Key::KEY_LEFTCTRL,
            Key::KEY_LEFTSHIFT,
            Key::KEY_LEFTALT,
            Key::KEY_TAB,
            Key::KEY_ENTER,
            Key::KEY_ESC,
            Key::KEY_SPACE,
            Key::KEY_BACKSPACE,
            Key::KEY_LEFT,
            Key::KEY_RIGHT,
            Key::KEY_UP,
            Key::KEY_DOWN,
            Key::KEY_PAGEUP,
            Key::KEY_PAGEDOWN,
            Key::KEY_HOME,
            Key::KEY_END,
            Key::KEY_PLAYPAUSE,
            Key::KEY_NEXTSONG,
            Key::KEY_PREVIOUSSONG,
            Key::KEY_MUTE,
            Key::KEY_VOLUMEUP,
            Key::KEY_VOLUMEDOWN,
            Key::KEY_F1,
            Key::KEY_F2,
            Key::KEY_F3,
            Key::KEY_F4,
            Key::KEY_F5,
            Key::KEY_F6,
            Key::KEY_F7,
            Key::KEY_F8,
            Key::KEY_F9,
            Key::KEY_F10,
            Key::KEY_F11,
            Key::KEY_F12,
            Key::KEY_0,
            Key::KEY_1,
            Key::KEY_2,
            Key::KEY_3,
            Key::KEY_4,
            Key::KEY_5,
            Key::KEY_6,
            Key::KEY_7,
            Key::KEY_8,
            Key::KEY_9,
            Key::KEY_A,
            Key::KEY_B,
            Key::KEY_C,
            Key::KEY_D,
            Key::KEY_E,
            Key::KEY_F,
            Key::KEY_G,
            Key::KEY_H,
            Key::KEY_I,
            Key::KEY_J,
            Key::KEY_K,
            Key::KEY_L,
            Key::KEY_M,
            Key::KEY_N,
            Key::KEY_O,
            Key::KEY_P,
            Key::KEY_Q,
            Key::KEY_R,
            Key::KEY_S,
            Key::KEY_T,
            Key::KEY_U,
            Key::KEY_V,
            Key::KEY_W,
            Key::KEY_X,
            Key::KEY_Y,
            Key::KEY_Z,
        ] {
            keys.insert(k);
        }

        let mut rel_axes = AttributeSet::<RelativeAxisType>::new();
        rel_axes.insert(RelativeAxisType::REL_X);
        rel_axes.insert(RelativeAxisType::REL_Y);
        rel_axes.insert(RelativeAxisType::REL_WHEEL);
        rel_axes.insert(RelativeAxisType::REL_HWHEEL);

        let abs_setup_x = UinputAbsSetup::new(
            AbsoluteAxisType::ABS_X,
            AbsInfo::new(0, 0, ABS_MAX, 0, 0, 1),
        );
        let abs_setup_y = UinputAbsSetup::new(
            AbsoluteAxisType::ABS_Y,
            AbsInfo::new(0, 0, ABS_MAX, 0, 0, 1),
        );

        let device = VirtualDeviceBuilder::new()
            .context("Failed to open /dev/uinput")?
            .name("gesture-esp-hid")
            .with_keys(&keys)
            .context("Failed to register uinput keys")?
            .with_relative_axes(&rel_axes)
            .context("Failed to register uinput relative axes")?
            .with_absolute_axis(&abs_setup_x)
            .context("Failed to register uinput ABS_X")?
            .with_absolute_axis(&abs_setup_y)
            .context("Failed to register uinput ABS_Y")?
            .build()
            .context("Failed to create uinput device")?;

        thread::sleep(Duration::from_millis(40));

        let has_wpctl = Command::new("wpctl")
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok();

        Ok(Self {
            device: Mutex::new(device),
            use_pactl_fallback: !has_wpctl,
        })
    }

    fn tap_key(&self, key: Key) -> Result<()> {
        let mut dev = self.device.lock();
        dev.emit(&[InputEvent::new(EventType::KEY, key.code(), 1)])?;
        dev.emit(&[InputEvent::new(EventType::KEY, key.code(), 0)])?;
        Ok(())
    }
}

impl OsController for LinuxUinputController {
    fn move_pointer_rel(&mut self, dx: i32, dy: i32) -> Result<()> {
        if dx == 0 && dy == 0 {
            return Ok(());
        }
        let events = [
            InputEvent::new(EventType::RELATIVE, RelativeAxisType::REL_X.0, dx),
            InputEvent::new(EventType::RELATIVE, RelativeAxisType::REL_Y.0, dy),
        ];
        self.device.lock().emit(&events)?;
        Ok(())
    }

    fn move_pointer_abs(&mut self, norm_x: f32, norm_y: f32) -> Result<()> {
        let ax = (norm_x.clamp(0.0, 1.0) * (ABS_MAX as f32)).round() as i32;
        let ay = (norm_y.clamp(0.0, 1.0) * (ABS_MAX as f32)).round() as i32;
        let events = [
            InputEvent::new(EventType::ABSOLUTE, AbsoluteAxisType::ABS_X.0, ax),
            InputEvent::new(EventType::ABSOLUTE, AbsoluteAxisType::ABS_Y.0, ay),
        ];
        self.device.lock().emit(&events)?;
        Ok(())
    }

    fn mouse_button(&mut self, button: MouseBtn, pressed: bool) -> Result<()> {
        let key = match button {
            MouseBtn::Left => Key::BTN_LEFT,
            MouseBtn::Right => Key::BTN_RIGHT,
            MouseBtn::Middle => Key::BTN_MIDDLE,
        };
        let val = if pressed { 1 } else { 0 };
        self.device
            .lock()
            .emit(&[InputEvent::new(EventType::KEY, key.code(), val)])?;
        Ok(())
    }

    fn mouse_scroll(&mut self, wheel_y: i32, wheel_x: i32) -> Result<()> {
        let mut events = Vec::with_capacity(2);
        if wheel_y != 0 {
            events.push(InputEvent::new(
                EventType::RELATIVE,
                RelativeAxisType::REL_WHEEL.0,
                wheel_y,
            ));
        }
        if wheel_x != 0 {
            events.push(InputEvent::new(
                EventType::RELATIVE,
                RelativeAxisType::REL_HWHEEL.0,
                wheel_x,
            ));
        }
        if !events.is_empty() {
            self.device.lock().emit(&events)?;
        }
        Ok(())
    }

    fn send_hotkey(&mut self, keys: &[VirtualKey]) -> Result<()> {
        let mapped: Vec<Key> = keys.iter().map(|&k| map_virtual_key_linux(k)).collect();
        if mapped.is_empty() {
            return Ok(());
        }

        let mut dev = self.device.lock();
        for &k in &mapped {
            dev.emit(&[InputEvent::new(EventType::KEY, k.code(), 1)])?;
        }
        thread::sleep(Duration::from_millis(8));
        for &k in mapped.iter().rev() {
            dev.emit(&[InputEvent::new(EventType::KEY, k.code(), 0)])?;
        }
        Ok(())
    }

    fn media_command(&mut self, cmd: MediaCmd) -> Result<()> {
        let key = match cmd {
            MediaCmd::PlayPause => Key::KEY_PLAYPAUSE,
            MediaCmd::Mute => Key::KEY_MUTE,
            MediaCmd::NextTrack => Key::KEY_NEXTSONG,
            MediaCmd::PrevTrack => Key::KEY_PREVIOUSSONG,
            MediaCmd::VolumeUp => Key::KEY_VOLUMEUP,
            MediaCmd::VolumeDown => Key::KEY_VOLUMEDOWN,
        };
        self.tap_key(key)
    }

    fn set_system_volume(&mut self, volume_0_to_1: f32) -> Result<()> {
        let pct = (volume_0_to_1.clamp(0.0, 1.0) * 100.0).round() as u32;
        if !self.use_pactl_fallback {
            let arg = format!("{pct}%");
            if Command::new("wpctl")
                .args(["set-volume", "@DEFAULT_AUDIO_SINK@", &arg])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .is_ok()
            {
                return Ok(());
            }
            self.use_pactl_fallback = true;
        }

        let arg = format!("{pct}%");
        Command::new("pactl")
            .args(["set-sink-volume", "@DEFAULT_SINK@", &arg])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to run wpctl or pactl")?;
        Ok(())
    }

    fn execute_shell(&mut self, command_line: &str) -> Result<()> {
        Command::new("sh")
            .args(["-c", command_line])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("Failed to spawn command: {command_line}"))?;
        Ok(())
    }
}

fn map_virtual_key_linux(vk: VirtualKey) -> Key {
    match vk {
        VirtualKey::Super => Key::KEY_LEFTMETA,
        VirtualKey::Ctrl => Key::KEY_LEFTCTRL,
        VirtualKey::Shift => Key::KEY_LEFTSHIFT,
        VirtualKey::Alt => Key::KEY_LEFTALT,
        VirtualKey::Tab => Key::KEY_TAB,
        VirtualKey::Enter => Key::KEY_ENTER,
        VirtualKey::Escape => Key::KEY_ESC,
        VirtualKey::Space => Key::KEY_SPACE,
        VirtualKey::Backspace => Key::KEY_BACKSPACE,
        VirtualKey::Left => Key::KEY_LEFT,
        VirtualKey::Right => Key::KEY_RIGHT,
        VirtualKey::Up => Key::KEY_UP,
        VirtualKey::Down => Key::KEY_DOWN,
        VirtualKey::PageUp => Key::KEY_PAGEUP,
        VirtualKey::PageDown => Key::KEY_PAGEDOWN,
        VirtualKey::Home => Key::KEY_HOME,
        VirtualKey::End => Key::KEY_END,
        VirtualKey::F(n) => match n {
            1 => Key::KEY_F1,
            2 => Key::KEY_F2,
            3 => Key::KEY_F3,
            4 => Key::KEY_F4,
            5 => Key::KEY_F5,
            6 => Key::KEY_F6,
            7 => Key::KEY_F7,
            8 => Key::KEY_F8,
            9 => Key::KEY_F9,
            10 => Key::KEY_F10,
            11 => Key::KEY_F11,
            _ => Key::KEY_F12,
        },
        VirtualKey::Digit(d) => match d {
            0 => Key::KEY_0,
            1 => Key::KEY_1,
            2 => Key::KEY_2,
            3 => Key::KEY_3,
            4 => Key::KEY_4,
            5 => Key::KEY_5,
            6 => Key::KEY_6,
            7 => Key::KEY_7,
            8 => Key::KEY_8,
            _ => Key::KEY_9,
        },
        VirtualKey::Alpha(ch) => match ch {
            'A' => Key::KEY_A,
            'B' => Key::KEY_B,
            'C' => Key::KEY_C,
            'D' => Key::KEY_D,
            'E' => Key::KEY_E,
            'F' => Key::KEY_F,
            'G' => Key::KEY_G,
            'H' => Key::KEY_H,
            'I' => Key::KEY_I,
            'J' => Key::KEY_J,
            'K' => Key::KEY_K,
            'L' => Key::KEY_L,
            'M' => Key::KEY_M,
            'N' => Key::KEY_N,
            'O' => Key::KEY_O,
            'P' => Key::KEY_P,
            'Q' => Key::KEY_Q,
            'R' => Key::KEY_R,
            'S' => Key::KEY_S,
            'T' => Key::KEY_T,
            'U' => Key::KEY_U,
            'V' => Key::KEY_V,
            'W' => Key::KEY_W,
            'X' => Key::KEY_X,
            'Y' => Key::KEY_Y,
            _ => Key::KEY_Z,
        },
    }
}
