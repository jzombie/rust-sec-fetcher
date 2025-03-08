use std::env;
use std::io::{self, IsTerminal};

/// Checks whether the program is running in an interactive terminal session.
///
/// This function first checks for the `INTERACTIVE_MODE` environment variable:
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
/// INTERACTIVE_MODE=0 cargo run
/// ```
pub fn is_interactive_mode() -> bool {
    if let Ok(value) = env::var("INTERACTIVE_MODE") {
        match value.as_str() {
            "1" | "true" => return true,   // Force interactive mode
            "0" | "false" => return false, // Force non-interactive mode
            _ => {} // Ignore invalid values and fallback to default behavior
        }
    }

    // Default behavior: check if stdin and stdout are terminals
    io::stdin().is_terminal() && io::stdout().is_terminal()
}
