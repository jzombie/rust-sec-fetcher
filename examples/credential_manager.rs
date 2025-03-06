use keyring::credential;
use sec_fetcher::network::{CredentialManager, CredentialProvider};
use std::io::{self, Write};

/// Prompt user for input
fn prompt_user(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = CredentialManager::from_prompt()?;

    let credential = manager.get_credential().unwrap();

    println!("credential: {:?}", credential);

    Ok(())
}
