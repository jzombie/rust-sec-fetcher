use std::io::{self, IsTerminal};

/// Checks whether the program is running in an interactive terminal session.
///
/// This function determines if both **standard input (stdin)** and **standard output (stdout)** 
/// are connected to a terminal, ensuring that the program is running in an **interactive mode**.
/// 
/// # Returns
/// - `true` if both stdin and stdout are attached to a terminal (i.e., an interactive shell).
/// - `false` if either stdin or stdout is redirected (e.g., piped input/output, running in a script).
///
/// # Examples
/// ```
/// use sec_fetcher::utils::is_interactive_mode;
///
/// if is_interactive_mode() {
///     println!("Running in interactive mode.");
/// } else {
///     println!("Not an interactive session.");
/// }
/// ```
///
/// # Notes
/// - This function is useful when determining whether user prompts or interactive behavior
///   should be enabled.
/// - If the program is run in a pipeline (e.g., `echo "data" | my_program`), this function
///   will return `false` because `stdin` is not connected to a terminal.
///
/// # Platform Compatibility
/// - **Unix & Linux:** Uses `isatty(3)` internally.
/// - **Windows:** Uses `GetConsoleMode` to check terminal status.
pub fn is_interactive_mode() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}
