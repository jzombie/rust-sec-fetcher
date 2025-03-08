use config::{Config, File};
use dirs::config_dir;
use std::error::Error;
use std::path::PathBuf;
use http_cache_reqwest::{Cache, CacheMode, CACacheManager, HttpCache, HttpCacheOptions};
use crate::config::{AppConfig, CredentialManager, CredentialProvider};
use crate::utils::is_terminal;

pub struct ConfigManager {
    config: AppConfig
}

impl ConfigManager {
    pub fn get_suggested_system_path() -> Option<PathBuf> {
        config_dir().map(|dir| dir.join(env!("CARGO_PKG_NAME")).join("config.toml"))
    }
    
    /// Determines the standard config file location.
    pub fn get_config_path() -> PathBuf {
        if let Some(path) = Self::get_suggested_system_path() {
            if path.exists() {
                return path;
            }
        }
        PathBuf::from("config.toml") // Fallback if no global config
    }
    
    // TODO: Add ability to set path directly in here (and/or add `from_config` method)
    /// Loads configuration from file and environment variables.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let config_path = Self::get_config_path();

        let config = Config::builder()
            .add_source(File::with_name(config_path.to_str().unwrap()).required(false)) // Load if exists
            .build()?;

        let mut settings: AppConfig = config.try_deserialize()?;
    
        if settings.email.is_none() {
            if is_terminal() {
                let credential_manager = CredentialManager::from_prompt()?;
                let email = credential_manager.get_credential()
                    .map_err(|err| format!("Could not obtain credential from credential manager: {:?}", err))?;
                settings.email = Some(email);
            } else {
                return Err("Could not obtain email credential".into());
            }
        }

        Ok(Self { config: settings })
    }

    /// Retrieves a reference to the configuration.
    pub fn get_config(&self) -> &AppConfig {
        &self.config
    }
}
