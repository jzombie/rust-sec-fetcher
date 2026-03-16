use crate::config::{AppConfig, CredentialManager, CredentialProvider};
use crate::utils::is_interactive_mode;
use crate::Caches;
use config::{Config, File};
use dirs::config_dir;
use merge::Merge;
use std::error::Error;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Environment variable used to supply the required contact email address.
///
/// ## Why email is required
///
/// The SEC mandates that all automated EDGAR data requests include a descriptive
/// `User-Agent` header containing a contact email address so that the SEC can
/// reach out if a client is causing problems.  See the SEC's official guidance:
/// <https://www.sec.gov/os/accessing-edgar-data>
///
/// Without a valid email address this library cannot construct a compliant
/// `User-Agent` header and will refuse to make network requests.
///
/// ## Privacy & security
///
/// The email address is included **only** in the `User-Agent` header that is
/// sent to SEC EDGAR servers.  It is not transmitted to any other party.
/// You are responsible for choosing the most secure method of supplying this
/// value that is appropriate for your own use case (e.g. an environment
/// variable, a config file with restricted permissions, a secrets manager,
/// etc.).
///
/// ## Disclaimer
///
/// This project is not affiliated with, endorsed by, or associated with the
/// U.S. Securities and Exchange Commission (SEC) in any way.
///
/// ## Precedence (highest → lowest)
///
/// 1. **Config file** — `email = "…"` key in `sec_fetcher_config.toml`
///    (or the path passed to [`ConfigManager::from_config`]).
/// 2. **This environment variable** — `SEC_FETCHER_EMAIL=your@example.com`
/// 3. **Interactive prompt** — when `stdin`/`stdout` are a terminal the user
///    is prompted at startup.
///
/// If none of the above supplies an address, [`ConfigManager::from_config`]
/// returns an error.
///
/// # Example
/// ```sh
/// SEC_FETCHER_EMAIL=your.name@example.com cargo run
/// ```
pub const EMAIL_ENV_VAR: &str = "SEC_FETCHER_EMAIL";

/// Environment variable used to override the app name in the `User-Agent` header.
///
/// ## Precedence (highest → lowest)
///
/// 1. **Config file** — `app_name = "…"` key in `sec_fetcher_config.toml`.
/// 2. **This environment variable** — `SEC_FETCHER_APP_NAME=my-app`
/// 3. **Hardcoded default** — `"sec-fetcher"`
///
/// # Example
/// ```sh
/// SEC_FETCHER_APP_NAME=my-app cargo run
/// ```
pub const APP_NAME_ENV_VAR: &str = "SEC_FETCHER_APP_NAME";

/// The default app name sent in the User-Agent header when neither the config
/// file nor `SEC_FETCHER_APP_NAME` supplies one.
pub const DEFAULT_APP_NAME: &str = "sec-fetcher";

/// Environment variable used to override the app version in the `User-Agent` header.
///
/// ## Precedence (highest → lowest)
///
/// 1. **Config file** — `app_version = "…"` key in `sec_fetcher_config.toml`.
/// 2. **This environment variable** — `SEC_FETCHER_APP_VERSION=1.2.3`
/// 3. **Hardcoded default** — the sec-fetcher crate version
///
/// # Example
/// ```sh
/// SEC_FETCHER_APP_VERSION=1.2.3 cargo run
/// ```
pub const APP_VERSION_ENV_VAR: &str = "SEC_FETCHER_APP_VERSION";

/// The default app version used in the User-Agent header when neither the config
/// file nor `SEC_FETCHER_APP_VERSION` supplies one.
pub const DEFAULT_APP_VERSION: &str = env!("CARGO_PKG_VERSION");

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
    /// ## Email resolution — precedence (highest → lowest)
    ///
    /// 1. **Config file** — `email = "…"` key in `sec_fetcher_config.toml`
    ///    (or the `path` argument to this function).
    /// 2. **Environment variable** — [`EMAIL_ENV_VAR`] (`SEC_FETCHER_EMAIL`).
    /// 3. **Interactive prompt** — when `stdin`/`stdout` are a terminal the
    ///    user is prompted at startup.
    ///
    /// Returns an error if none of the above provides an address.
    ///
    /// ## App name resolution — precedence (highest → lowest)
    ///
    /// 1. **Config file** — `app_name = "…"` key.
    /// 2. **Environment variable** — [`APP_NAME_ENV_VAR`] (`SEC_FETCHER_APP_NAME`).
    /// 3. **Default** — [`DEFAULT_APP_NAME`] (`"sec-fetcher"`).
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
            if let Ok(email) = std::env::var(EMAIL_ENV_VAR) {
                user_settings.email = Some(email);
            } else if is_interactive_mode() {
                let credential_manager = CredentialManager::from_prompt()?;
                let email = credential_manager.get_credential().map_err(|err| {
                    format!(
                        "Could not obtain credential from credential manager: {:?}",
                        err
                    )
                })?;
                user_settings.email = Some(email);
            } else {
                return Err(format!(
                    "Could not obtain email credential. Set `email` in the config file or the `{}` environment variable.",
                    EMAIL_ENV_VAR
                ).into());
            }
        }

        settings.merge(user_settings);

        // Resolve app_name: env var fills the gap when neither TOML nor a
        // direct AppConfig assignment supplied one.
        if settings.app_name.is_none() {
            if let Ok(name) = std::env::var(APP_NAME_ENV_VAR) {
                settings.app_name = Some(name);
            }
        }

        // Resolve app_version: env var fills the gap when neither TOML nor a
        // direct AppConfig assignment supplied one.
        if settings.app_version.is_none() {
            if let Ok(version) = std::env::var(APP_VERSION_ENV_VAR) {
                settings.app_version = Some(version);
            }
        }

        let instance = Self { config: settings };

        instance.init_caches();

        Ok(instance)
    }

    pub fn from_app_config(app_config: &AppConfig) -> Self {
        let instance = Self {
            config: app_config.clone(),
        };

        instance.init_caches();

        instance
    }

    fn init_caches(&self) {
        Caches::init(self)
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
