use crate::config::{AppConfig, CredentialManager, CredentialProvider};
use crate::utils::is_interactive_mode;
use config::{Config, File};
use dirs::config_dir;
use merge::Merge;
use std::error::Error;
use std::path::PathBuf;
use std::sync::LazyLock;

#[derive(Debug)]
pub struct ConfigManager {
    config: AppConfig,
}

static DEFAULT_CONFIG_PATH: LazyLock<String> =
    LazyLock::new(|| format!("{}_config.toml", env!("CARGO_PKG_NAME").replace("-", "_")));

impl ConfigManager {
    /// Loads configuration using the default path.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        Self::from_config(None)
    }

    /// Loads configuration from the given file path (if provided) or defaults to the standard config path.
    ///
    /// If no path is provided, the default configuration path will be used.
    pub fn from_config(path: Option<PathBuf>) -> Result<Self, Box<dyn Error>> {
        if let Some(path) = &path {
            if !path.exists() {
                return Err(format!(
                    "Config path does not exist: {}",
                    path.to_string_lossy().into_owned()
                )
                .into());
            }
        };

        let config_path = path.unwrap_or_else(Self::get_config_path);

        let config = Config::builder()
            .add_source(File::with_name(config_path.to_str().unwrap()).required(false)) // Load if exists
            .build()?;

        let mut settings: AppConfig = AppConfig::default();
        let mut user_settings: AppConfig =
            config
                .try_deserialize()
                .map_err(|err| -> Box<dyn std::error::Error> {
                    let valid_keys_list = AppConfig::get_valid_keys()
                        .iter()
                        .map(|(key, typ)| format!("  - {} ({})", key, typ))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let error_message = format!(
                        "{}\n\nValid configuration keys are:\n{}",
                        err, valid_keys_list
                    );

                    error_message.into()
                })?;

        if settings.email.is_none() && user_settings.email.is_none() {
            if is_interactive_mode() {
                let credential_manager = CredentialManager::from_prompt()?;
                let email = credential_manager.get_credential().map_err(|err| {
                    format!(
                        "Could not obtain credential from credential manager: {:?}",
                        err
                    )
                })?;
                user_settings.email = Some(email);
            } else {
                return Err("Could not obtain email credential".into());
            }
        }

        settings.merge(user_settings);

        Ok(Self { config: settings })
    }

    pub fn from_app_config(app_config: &AppConfig) -> Self {
        Self {
            config: app_config.clone(),
        }
    }

    /// Retrieves a reference to the configuration.
    pub fn get_config(&self) -> &AppConfig {
        &self.config
    }

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
        PathBuf::from(&*DEFAULT_CONFIG_PATH)
    }
}
