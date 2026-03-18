use crate::caches::Caches;
use crate::config::AppConfig;
#[cfg(feature = "keyring")]
use crate::config::{CredentialManager, CredentialProvider};
#[cfg(feature = "keyring")]
use crate::utils::is_interactive_mode;
use config::{Config, File};
use dirs::config_dir;
use merge::Merge;
use simd_r_drive::DataStore;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

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
    caches: Caches,
    /// Keeps the temporary cache directory alive for the lifetime of this
    /// `ConfigManager`.  `None` when `cache_base_dir` was explicitly configured.
    _cache_dir: Option<tempfile::TempDir>,
}

static DEFAULT_CONFIG_PATH: LazyLock<String> =
    LazyLock::new(|| format!("{}_config.toml", env!("CARGO_PKG_NAME").replace("-", "_")));

impl ConfigManager {
    /// Loads configuration using the default path.
    pub fn load() -> Result<Self, Box<dyn Error>> {
        Self::from_config_with_app_identity(None, None, None)
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
        Self::from_config_with_app_identity(path, None, None)
    }

    /// Loads configuration and applies optional app identity string overrides.
    ///
    /// `app_name_override` and `app_version_override` have the **highest**
    /// precedence for `User-Agent` identity fields.
    ///
    /// ## App name resolution — precedence (highest → lowest)
    ///
    /// 1. **Function argument** — `app_name_override`.
    /// 2. **Config file** — `app_name = "…"` key.
    /// 3. **Environment variable** — [`APP_NAME_ENV_VAR`] (`SEC_FETCHER_APP_NAME`).
    /// 4. **Default** — [`DEFAULT_APP_NAME`] (`"sec-fetcher"`).
    ///
    /// ## App version resolution — precedence (highest → lowest)
    ///
    /// 1. **Function argument** — `app_version_override`.
    /// 2. **Config file** — `app_version = "…"` key.
    /// 3. **Environment variable** — [`APP_VERSION_ENV_VAR`] (`SEC_FETCHER_APP_VERSION`).
    /// 4. **Default** — [`DEFAULT_APP_VERSION`].
    pub fn from_config_with_app_identity(
        path: Option<PathBuf>,
        app_name_override: Option<&str>,
        app_version_override: Option<&str>,
    ) -> Result<Self, Box<dyn Error>> {
        if let Some(path) = &path
            && !path.exists()
        {
            return Err(format!(
                "Config path does not exist: {}",
                path.to_string_lossy().into_owned()
            )
            .into());
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
            } else {
                #[cfg(feature = "keyring")]
                if is_interactive_mode() {
                    let credential_manager = CredentialManager::from_prompt()?;
                    let email = credential_manager.get_credential().map_err(|err| {
                        format!(
                            "Could not obtain credential from credential manager: {:?}",
                            err
                        )
                    })?;
                    user_settings.email = Some(email);
                }
            }
            // Without the keyring feature the interactive prompt is not available;
            // the error branch below handles the no-email case.
            if user_settings.email.is_none() {
                return Err(format!(
                    "Could not obtain email credential. Set `email` in the config file or the `{}` environment variable.",
                    EMAIL_ENV_VAR
                ).into());
            }
        }

        settings.merge(user_settings);

        // Resolve app_name: env var fills the gap when neither TOML nor a
        // direct AppConfig assignment supplied one.
        if settings.app_name.is_none()
            && let Ok(name) = std::env::var(APP_NAME_ENV_VAR)
        {
            settings.app_name = Some(name);
        }

        // Resolve app_version: env var fills the gap when neither TOML nor a
        // direct AppConfig assignment supplied one.
        if settings.app_version.is_none()
            && let Ok(version) = std::env::var(APP_VERSION_ENV_VAR)
        {
            settings.app_version = Some(version);
        }

        if let Some(name) = app_name_override {
            settings.app_name = Some(name.to_string());
        }

        if let Some(version) = app_version_override {
            settings.app_version = Some(version.to_string());
        }

        let (caches, temp_dir) = Self::make_caches(&settings);

        let instance = Self {
            config: settings,
            caches,
            _cache_dir: temp_dir,
        };

        Ok(instance)
    }

    pub fn from_app_config(app_config: &AppConfig) -> Self {
        let (caches, temp_dir) = Self::make_caches(app_config);
        Self {
            config: app_config.clone(),
            caches,
            _cache_dir: temp_dir,
        }
    }

    /// Creates `Caches` for the given config.  When `cache_base_dir` is
    /// `None` a fresh temporary directory is created and returned so the
    /// caller can hold it alive.
    fn make_caches(config: &AppConfig) -> (Caches, Option<tempfile::TempDir>) {
        match &config.cache_base_dir {
            Some(path) => {
                let caches = Caches::open(path).unwrap_or_else(|err| {
                    panic!("Failed to open caches at '{}': {err}", path.display())
                });
                (caches, None)
            }
            None => {
                let temp_dir = tempfile::Builder::new()
                    .prefix(env!("CARGO_PKG_NAME"))
                    .tempdir()
                    .expect("failed to create temp cache dir");
                let caches = Caches::open(temp_dir.path())
                    .unwrap_or_else(|err| panic!("Failed to open caches in tempdir: {err}"));
                (caches, Some(temp_dir))
            }
        }
    }

    /// Retrieves a reference to the configuration.
    pub fn get_config(&self) -> &AppConfig {
        &self.config
    }

    pub fn get_http_cache_store(&self) -> Arc<DataStore> {
        self.caches.get_http_cache_store()
    }

    pub fn get_preprocessor_cache(&self) -> Arc<DataStore> {
        self.caches.get_preprocessor_cache()
    }

    pub fn get_suggested_system_path() -> Option<PathBuf> {
        config_dir().map(|dir| dir.join(env!("CARGO_PKG_NAME")).join("config.toml"))
    }

    /// Determines the standard config file location.
    pub fn get_config_path() -> PathBuf {
        if let Some(path) = Self::get_suggested_system_path()
            && path.exists()
        {
            return path;
        }
        PathBuf::from(&*DEFAULT_CONFIG_PATH)
    }
}
