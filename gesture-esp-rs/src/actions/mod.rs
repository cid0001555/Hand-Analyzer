pub mod dispatcher;

#[cfg(target_os = "linux")]
pub mod linux_uinput;

#[cfg(target_os = "windows")]
pub mod windows_win32;

use anyhow::Result;
use crate::fsm::{MediaCmd, MouseBtn, VirtualKey};

pub use dispatcher::spawn_action_dispatcher_thread;

pub trait OsController: Send + Sync {
    fn move_pointer_rel(&mut self, dx: i32, dy: i32) -> Result<()>;
    fn move_pointer_abs(&mut self, norm_x: f32, norm_y: f32) -> Result<()>;
    fn mouse_button(&mut self, button: MouseBtn, pressed: bool) -> Result<()>;
    fn mouse_scroll(&mut self, wheel_y: i32, wheel_x: i32) -> Result<()>;
    fn send_hotkey(&mut self, keys: &[VirtualKey]) -> Result<()>;
    fn media_command(&mut self, cmd: MediaCmd) -> Result<()>;
    fn set_system_volume(&mut self, volume_0_to_1: f32) -> Result<()>;
    fn execute_shell(&mut self, command_line: &str) -> Result<()>;
}

pub fn create_os_controller() -> Result<Box<dyn OsController>> {
    #[cfg(target_os = "linux")]
    {
        let ctrl = linux_uinput::LinuxUinputController::new()?;
        Ok(Box::new(ctrl))
    }

    #[cfg(target_os = "windows")]
    {
        let ctrl = windows_win32::WindowsInputController::new()?;
        Ok(Box::new(ctrl))
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Ok(Box::new(NullOsController))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
struct NullOsController;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
impl OsController for NullOsController {
    fn move_pointer_rel(&mut self, _dx: i32, _dy: i32) -> Result<()> {
        Ok(())
    }
    fn move_pointer_abs(&mut self, _norm_x: f32, _norm_y: f32) -> Result<()> {
        Ok(())
    }
    fn mouse_button(&mut self, _button: MouseBtn, _pressed: bool) -> Result<()> {
        Ok(())
    }
    fn mouse_scroll(&mut self, _wheel_y: i32, _wheel_x: i32) -> Result<()> {
        Ok(())
    }
    fn send_hotkey(&mut self, _keys: &[VirtualKey]) -> Result<()> {
        Ok(())
    }
    fn media_command(&mut self, _cmd: MediaCmd) -> Result<()> {
        Ok(())
    }
    fn set_system_volume(&mut self, _volume_0_to_1: f32) -> Result<()> {
        Ok(())
    }
    fn execute_shell(&mut self, command_line: &str) -> Result<()> {
        use std::process::{Command, Stdio};
        Command::new("sh")
            .args(["-c", command_line])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        Ok(())
    }
}
