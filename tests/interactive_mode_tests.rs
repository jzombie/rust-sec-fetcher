use std::env;
use sec_fetcher::utils::is_interactive_mode;

#[test]
fn test_interactive_mode_override() {
    env::set_var("INTERACTIVE_MODE", "true");
    assert!(is_interactive_mode()); // Forced interactive mode

    env::set_var("INTERACTIVE_MODE", "1");
    assert!(is_interactive_mode()); // Forced interactive mode

    env::set_var("INTERACTIVE_MODE", "false");
    assert!(!is_interactive_mode()); // Forced non-interactive mode

    env::set_var("INTERACTIVE_MODE", "0");
    assert!(!is_interactive_mode()); // Forced non-interactive mode

    env::remove_var("INTERACTIVE_MODE"); // Restore original behavior
}
