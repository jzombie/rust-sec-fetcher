use keyring::Entry;
use std::error::Error;
use std::io::{self, Write};
use crate::utils::is_interactive_mode;

/// Service name used for keyring storage
const SERVICE: &str = env!("CARGO_PKG_NAME");

/// Credential Manager struct
pub struct CredentialManager {
    entry: Entry,
    cached: Option<String>,
}

pub trait CredentialProvider {
    fn from_prompt() -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}

impl CredentialManager {
    /// Creates a new instance for a given user
    pub fn new(username: &str) -> Result<Self, Box<dyn Error>> {
        let entry = Entry::new(SERVICE, username)?;
        let instance = Self {
            entry,
            cached: None,
        };

        Ok(instance)
    }

    /// Stores a credential securely
    pub fn store_credential(&mut self, credential: &str) -> Result<(), Box<dyn Error>> {
        self.cached = Some(credential.to_string());

        self.entry.set_password(credential)?;

        Ok(())
    }

    /// Retrieves stored credential
    pub fn get_credential(&self) -> Result<String, Box<dyn Error>> {
        match &self.cached {
            Some(credential) => Ok(credential.clone()), // Return cached credential
            None => {
                // Attempt to retrieve credential and propagate error if it fails
                let credential = self.entry.get_password()?;
                Ok(credential)
            }
        }
    }

    /// Deletes stored credential
    pub fn delete_credential(&self) -> Result<(), Box<dyn Error>> {
        // TODO: Is this the way?
        if self.entry.set_password("").is_ok() {
            println!("Credential deleted.");
        } else {
            return Err("Failed to delete credential.".into());
        }

        Ok(())
    }
}

impl CredentialProvider for CredentialManager {
    /// Gets the credential or prompts the user for input
    fn from_prompt() -> Result<Self, Box<dyn Error>> {
        if !is_interactive_mode() {
            return Err("`from_prompt` can only be run in interactive terminal mode.".into());
        }

        print!("Enter your username: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let username = input.trim().to_string();

        let mut instance = Self::new(&username)?;

        match instance.get_credential() {
            Ok(existing) => {
                println!("Stored credential found");

                println!("TODO: Remove: {}", existing);

                existing
            }
            _ => {
                print!("Enter your email: ");
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let credential = input.trim().to_string();

                if let Err(err) = instance.store_credential(&credential) {
                    if let Some(cached_credential) = instance.get_credential().ok() {
                        if cached_credential == credential {
                            eprintln!("Using cached credential for this session, but caught the following error: {}", err);
                        }
                    }
                }

                credential
            }
        };

        Ok(instance)
    }
}
