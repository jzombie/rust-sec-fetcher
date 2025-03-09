use sec_fetcher::utils::{is_interactive_mode, set_interactive_mode_override};
use std::env;

#[test]
fn test_interactive_mode_override() {
    assert!(env::var("INTERACTIVE_MODE_OVERRIDE").is_err()); // Check if unset

    set_interactive_mode_override(Some(true));
    assert!(is_interactive_mode());
    assert_eq!(env::var("INTERACTIVE_MODE_OVERRIDE").unwrap(), "1");

    set_interactive_mode_override(Some(false));
    assert!(!is_interactive_mode());
    assert_eq!(env::var("INTERACTIVE_MODE_OVERRIDE").unwrap(), "0");

    set_interactive_mode_override(None); // Restore original behavior
    assert!(env::var("INTERACTIVE_MODE_OVERRIDE").is_err());
}
