use std::env;
use std::io::{self, IsTerminal};

static INTERACTIVE_MODE_OVERRIDE_KEY: &str = "INTERACTIVE_MODE_OVERRIDE";

/// Checks whether the program is running in an interactive terminal session.
///
/// This function first checks for the `INTERACTIVE_MODE_OVERRIDE` environment variable:
/// - If set to `"1"` or `"true"`, it **forces interactive mode**.
/// - If set to `"0"` or `"false"`, it **forces non-interactive mode**.
/// - Otherwise, it falls back to checking if `stdin` and `stdout` are terminals.
///
/// # Returns
/// - `true` if interactive mode is detected or forced.
/// - `false` if running in a pipeline, script, or if overridden.
///
/// # Example Override
/// ```sh
/// INTERACTIVE_MODE_OVERRIDE=0 cargo run
/// ```
pub fn is_interactive_mode() -> bool {
    if let Ok(value) = env::var(INTERACTIVE_MODE_OVERRIDE_KEY) {
        match value.as_str() {
            "1" | "true" => return true,   // Force interactive mode
            "0" | "false" => return false, // Force non-interactive mode
            _ => {} // Ignore invalid values and fallback to default behavior
        }
    }

    // Default behavior: check if stdin and stdout are terminals
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub fn set_interactive_mode_override(mode_override: Option<bool>) {
    match mode_override {
        Some(mode_override) => {
            env::set_var(INTERACTIVE_MODE_OVERRIDE_KEY, match mode_override {
                true => "1",
                false =>"0"
            })
        },
        None => env::remove_var(INTERACTIVE_MODE_OVERRIDE_KEY)
    }
}
