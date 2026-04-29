/// Unit tests for [`sec_fetcher::utils::is_interactive_mode`] and
/// [`sec_fetcher::utils::set_interactive_mode_override`].
use sec_fetcher::utils::{is_interactive_mode, set_interactive_mode_override};
use std::io::IsTerminal;
use std::sync::Mutex;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn test_override_true_forces_interactive() {
    let _lock = ENV_MUTEX.lock().unwrap();
    set_interactive_mode_override(Some(true));
    assert!(is_interactive_mode());
    set_interactive_mode_override(None);
}

#[test]
fn test_override_false_forces_non_interactive() {
    let _lock = ENV_MUTEX.lock().unwrap();
    set_interactive_mode_override(Some(false));
    assert!(!is_interactive_mode());
    set_interactive_mode_override(None);
}

#[test]
fn test_override_true_then_false() {
    let _lock = ENV_MUTEX.lock().unwrap();
    set_interactive_mode_override(Some(true));
    assert!(is_interactive_mode());

    set_interactive_mode_override(Some(false));
    assert!(!is_interactive_mode());

    set_interactive_mode_override(None);
}

#[test]
fn test_override_clear_restores_terminal_detection() {
    let _lock = ENV_MUTEX.lock().unwrap();
    // Set then clear
    set_interactive_mode_override(Some(true));
    assert!(is_interactive_mode());
    set_interactive_mode_override(None);

    // After clearing, behavior should depend on whether we're in a terminal.
    // Since we're running tests non-interactively, stdin is likely not a terminal,
    // so is_interactive_mode is expected to be false.
    let after_clear = is_interactive_mode();
    let cin = std::io::stdin();
    let cout = std::io::stdout();
    let expected = cin.is_terminal() && cout.is_terminal();
    assert_eq!(after_clear, expected);
}

#[test]
fn test_env_var_override_interactive() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe {
        std::env::set_var("INTERACTIVE_MODE_OVERRIDE", "1");
    }
    assert!(is_interactive_mode());
    unsafe {
        std::env::remove_var("INTERACTIVE_MODE_OVERRIDE");
    }
}

#[test]
fn test_env_var_override_non_interactive() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe {
        std::env::set_var("INTERACTIVE_MODE_OVERRIDE", "0");
    }
    assert!(!is_interactive_mode());
    unsafe {
        std::env::remove_var("INTERACTIVE_MODE_OVERRIDE");
    }
}

#[test]
fn test_env_var_override_true_string() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe {
        std::env::set_var("INTERACTIVE_MODE_OVERRIDE", "true");
    }
    assert!(is_interactive_mode());
    unsafe {
        std::env::remove_var("INTERACTIVE_MODE_OVERRIDE");
    }
}

#[test]
fn test_env_var_override_false_string() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe {
        std::env::set_var("INTERACTIVE_MODE_OVERRIDE", "false");
    }
    assert!(!is_interactive_mode());
    unsafe {
        std::env::remove_var("INTERACTIVE_MODE_OVERRIDE");
    }
}

#[test]
fn test_env_var_invalid_value_falls_back() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe {
        std::env::set_var("INTERACTIVE_MODE_OVERRIDE", "invalid");
    }
    // Falls back to terminal detection (may be true or false depending on environment).
    // Just verify it doesn't panic and the env var is removed afterwards.
    let _result = is_interactive_mode();
    unsafe {
        std::env::remove_var("INTERACTIVE_MODE_OVERRIDE");
    }
}

#[test]
fn test_set_interactive_mode_override_none_removes_var() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe {
        std::env::set_var("INTERACTIVE_MODE_OVERRIDE", "1");
    }
    assert!(std::env::var("INTERACTIVE_MODE_OVERRIDE").is_ok());

    set_interactive_mode_override(None);
    assert!(std::env::var("INTERACTIVE_MODE_OVERRIDE").is_err());
}
