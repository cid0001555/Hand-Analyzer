use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::Receiver;
use tracing::{debug, error, info, warn};

use crate::actions::{create_os_controller, OsController};
use crate::fsm::OsActionEvent;

pub fn spawn_action_dispatcher_thread(
    action_rx: Receiver<OsActionEvent>,
    shutdown: Arc<AtomicBool>,
    dry_run: bool,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("action-dispatcher".to_string())
        .spawn(move || {
            let mut controller: Option<Box<dyn OsController>> = if dry_run {
                info!("Dry-run mode enabled");
                None
            } else {
                match create_os_controller() {
                    Ok(ctrl) => {
                        info!("Input backend ready");
                        Some(ctrl)
                    }
                    Err(err) => {
                        error!(error = %err, "Failed to initialize input controller");
                        None
                    }
                }
            };

            while !shutdown.load(Ordering::Relaxed) {
                let event = match action_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(ev) => ev,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                };

                debug!(?event, "Dispatching action");

                let Some(ctrl) = controller.as_mut() else {
                    continue;
                };

                let res = match event {
                    OsActionEvent::MovePointerRel { dx, dy } => ctrl.move_pointer_rel(dx, dy),
                    OsActionEvent::MovePointerAbs { norm_x, norm_y } => {
                        ctrl.move_pointer_abs(norm_x, norm_y)
                    }
                    OsActionEvent::MouseButton { button, pressed } => {
                        ctrl.mouse_button(button, pressed)
                    }
                    OsActionEvent::MouseScroll { wheel_y, wheel_x } => {
                        ctrl.mouse_scroll(wheel_y, wheel_x)
                    }
                    OsActionEvent::Hotkey { ref keys } => ctrl.send_hotkey(keys),
                    OsActionEvent::Media(cmd) => ctrl.media_command(cmd),
                    OsActionEvent::SetSystemVolume(vol) => ctrl.set_system_volume(vol),
                    OsActionEvent::ExecuteShell(ref cmdline) => ctrl.execute_shell(cmdline),
                };

                if let Err(err) = res {
                    warn!(error = %err, "Action execution failed");
                }
            }
        })
        .expect("Failed to spawn action dispatcher thread")
}
