#![cfg(target_os = "windows")]

use std::cell::RefCell;
use std::mem::size_of;
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
use windows::Win32::Media::Audio::{
    eConsole, eRender, IMMDeviceEnumerator, MMDeviceEnumerator,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_HWHEEL, MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_MOVE,
    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_WHEEL,
    MOUSEINPUT, VIRTUAL_KEY, VK_BACK, VK_DOWN, VK_END, VK_ESCAPE, VK_F1, VK_HOME, VK_LCONTROL,
    VK_LEFT, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_MEDIA_NEXT_TRACK, VK_MEDIA_PLAY_PAUSE,
    VK_MEDIA_PREV_TRACK, VK_NEXT, VK_PRIOR, VK_RETURN, VK_RIGHT, VK_SPACE, VK_TAB, VK_UP,
    VK_VOLUME_DOWN, VK_VOLUME_MUTE, VK_VOLUME_UP,
};

use crate::actions::OsController;
use crate::fsm::{MediaCmd, MouseBtn, VirtualKey};

const WHEEL_DELTA_WIN32: i32 = 120;

thread_local! {
    static THREAD_AUDIO_ENDPOINT: RefCell<Option<IAudioEndpointVolume>> = const { RefCell::new(None) };
}

pub struct WindowsInputController;

impl WindowsInputController {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    fn dispatch_inputs(inputs: &[INPUT]) -> Result<()> {
        if inputs.is_empty() {
            return Ok(());
        }
        let sent = unsafe { SendInput(inputs, size_of::<INPUT>() as i32) };
        if sent == 0 {
            return Err(anyhow!("SendInput sent 0 events"));
        }
        Ok(())
    }

    fn build_key_input(vk: VIRTUAL_KEY, is_extended: bool, key_up: bool) -> INPUT {
        let mut flags = Default::default();
        if is_extended {
            flags |= KEYEVENTF_EXTENDEDKEY;
        }
        if key_up {
            flags |= KEYEVENTF_KEYUP;
        }
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }
}

impl OsController for WindowsInputController {
    fn move_pointer_rel(&mut self, dx: i32, dy: i32) -> Result<()> {
        if dx == 0 && dy == 0 {
            return Ok(());
        }
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx,
                    dy,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_MOVE,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        Self::dispatch_inputs(&[input])
    }

    fn move_pointer_abs(&mut self, norm_x: f32, norm_y: f32) -> Result<()> {
        let ax = (norm_x.clamp(0.0, 1.0) * 65535.0).round() as i32;
        let ay = (norm_y.clamp(0.0, 1.0) * 65535.0).round() as i32;
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: ax,
                    dy: ay,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        Self::dispatch_inputs(&[input])
    }

    fn mouse_button(&mut self, button: MouseBtn, pressed: bool) -> Result<()> {
        let flags = match (button, pressed) {
            (MouseBtn::Left, true) => MOUSEEVENTF_LEFTDOWN,
            (MouseBtn::Left, false) => MOUSEEVENTF_LEFTUP,
            (MouseBtn::Right, true) => MOUSEEVENTF_RIGHTDOWN,
            (MouseBtn::Right, false) => MOUSEEVENTF_RIGHTUP,
            (MouseBtn::Middle, true) => MOUSEEVENTF_MIDDLEDOWN,
            (MouseBtn::Middle, false) => MOUSEEVENTF_MIDDLEUP,
        };
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        Self::dispatch_inputs(&[input])
    }

    fn mouse_scroll(&mut self, wheel_y: i32, wheel_x: i32) -> Result<()> {
        let mut inputs = Vec::with_capacity(2);
        if wheel_y != 0 {
            inputs.push(INPUT {
                r#type: INPUT_MOUSE,
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx: 0,
                        dy: 0,
                        mouseData: (wheel_y * WHEEL_DELTA_WIN32) as u32,
                        dwFlags: MOUSEEVENTF_WHEEL,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            });
        }
        if wheel_x != 0 {
            inputs.push(INPUT {
                r#type: INPUT_MOUSE,
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx: 0,
                        dy: 0,
                        mouseData: (wheel_x * WHEEL_DELTA_WIN32) as u32,
                        dwFlags: MOUSEEVENTF_HWHEEL,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            });
        }
        Self::dispatch_inputs(&inputs)
    }

    fn send_hotkey(&mut self, keys: &[VirtualKey]) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let mapped: Vec<(VIRTUAL_KEY, bool)> =
            keys.iter().map(|&k| map_virtual_key_win32(k)).collect();

        let mut inputs = Vec::with_capacity(mapped.len() * 2);
        for &(vk, ext) in &mapped {
            inputs.push(Self::build_key_input(vk, ext, false));
        }
        for &(vk, ext) in mapped.iter().rev() {
            inputs.push(Self::build_key_input(vk, ext, true));
        }
        Self::dispatch_inputs(&inputs)
    }

    fn media_command(&mut self, cmd: MediaCmd) -> Result<()> {
        let vk = match cmd {
            MediaCmd::PlayPause => VK_MEDIA_PLAY_PAUSE,
            MediaCmd::Mute => VK_VOLUME_MUTE,
            MediaCmd::NextTrack => VK_MEDIA_NEXT_TRACK,
            MediaCmd::PrevTrack => VK_MEDIA_PREV_TRACK,
            MediaCmd::VolumeUp => VK_VOLUME_UP,
            MediaCmd::VolumeDown => VK_VOLUME_DOWN,
        };
        let inputs = [
            Self::build_key_input(vk, true, false),
            Self::build_key_input(vk, true, true),
        ];
        Self::dispatch_inputs(&inputs)
    }

    fn set_system_volume(&mut self, volume_0_to_1: f32) -> Result<()> {
        let scalar = volume_0_to_1.clamp(0.0, 1.0);

        THREAD_AUDIO_ENDPOINT.with(|slot| -> Result<()> {
            let mut guard = slot.borrow_mut();
            if guard.is_none() {
                *guard = Some(acquire_audio_endpoint()?);
            }

            if let Some(ep) = guard.as_ref() {
                let call_res = unsafe { ep.SetMasterVolumeLevelScalar(scalar, std::ptr::null()) };
                if call_res.is_ok() {
                    return Ok(());
                }
            }

            let fresh_ep = acquire_audio_endpoint()?;
            unsafe {
                fresh_ep
                    .SetMasterVolumeLevelScalar(scalar, std::ptr::null())
                    .context("SetMasterVolumeLevelScalar failed")?;
            }
            *guard = Some(fresh_ep);
            Ok(())
        })
    }

    fn execute_shell(&mut self, command_line: &str) -> Result<()> {
        Command::new("cmd")
            .args(["/C", command_line])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("Failed to spawn cmd /C '{command_line}'"))?;
        Ok(())
    }
}

fn acquire_audio_endpoint() -> Result<IAudioEndpointVolume> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .context("CoCreateInstance failed")?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .context("GetDefaultAudioEndpoint failed")?;
        let endpoint: IAudioEndpointVolume = device
            .Activate(CLSCTX_ALL, None)
            .context("Activate(IAudioEndpointVolume) failed")?;
        Ok(endpoint)
    }
}

fn map_virtual_key_win32(vk: VirtualKey) -> (VIRTUAL_KEY, bool) {
    match vk {
        VirtualKey::Super => (VK_LWIN, true),
        VirtualKey::Ctrl => (VK_LCONTROL, false),
        VirtualKey::Shift => (VK_LSHIFT, false),
        VirtualKey::Alt => (VK_LMENU, false),
        VirtualKey::Tab => (VK_TAB, false),
        VirtualKey::Enter => (VK_RETURN, false),
        VirtualKey::Escape => (VK_ESCAPE, false),
        VirtualKey::Space => (VK_SPACE, false),
        VirtualKey::Backspace => (VK_BACK, false),
        VirtualKey::Left => (VK_LEFT, true),
        VirtualKey::Right => (VK_RIGHT, true),
        VirtualKey::Up => (VK_UP, true),
        VirtualKey::Down => (VK_DOWN, true),
        VirtualKey::PageUp => (VK_PRIOR, true),
        VirtualKey::PageDown => (VK_NEXT, true),
        VirtualKey::Home => (VK_HOME, true),
        VirtualKey::End => (VK_END, true),
        VirtualKey::F(n) => {
            let offset = (n.clamp(1, 12) - 1) as u16;
            (VIRTUAL_KEY(VK_F1.0 + offset), false)
        }
        VirtualKey::Alpha(ch) => (VIRTUAL_KEY(ch as u16), false),
        VirtualKey::Digit(d) => (VIRTUAL_KEY((b'0' + d.min(9)) as u16), false),
    }
}
