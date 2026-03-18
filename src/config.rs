mod app_config;
pub use app_config::AppConfig;

mod config_manager;
pub use config_manager::{
    ConfigManager, APP_NAME_ENV_VAR, APP_VERSION_ENV_VAR, DEFAULT_APP_NAME, DEFAULT_APP_VERSION,
    EMAIL_ENV_VAR,
};

mod credential_manager;
#[cfg(feature = "keyring")]
pub use credential_manager::{CredentialManager, CredentialProvider};
