use std::env;
use serde::{Serialize, Deserialize};
use serde_json::to_string_pretty;
use std::path::PathBuf;
use http_cache_reqwest::{Cache, CacheMode, CACacheManager, HttpCache, HttpCacheOptions};


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub email: Option<String>,
    pub max_concurrent: Option<usize>,
    pub min_delay_ms: Option<u64>,
    pub max_retries: Option<usize>,
    pub cache_dir: Option<String>, // TODO: Use PathBuf directly
    pub cache_mode: Option<String>, // TODO: Use CacheMode directly
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            email: None,
            max_concurrent: Some(1),
            min_delay_ms: Some(1000),
            max_retries: Some(5),
            // TODO: Differentiate between network cache and transformed asset cache
            cache_dir: None,
            cache_mode: None
        }
    }
}

impl AppConfig {
    pub fn pretty_print(&self) -> String {
        to_string_pretty(self).unwrap_or_else(|_| "Failed to serialize config".to_string())
    }

    /// Returns the cache directory path.
    ///
    /// If a cache directory is explicitly configured, it returns that path.
    /// Otherwise, it defaults to a subdirectory within the system's temporary directory,
    /// using the package name (from Cargo) to avoid conflicts.
    ///
    /// # Returns
    /// - `PathBuf` pointing to the cache directory.
    pub fn get_cache_dir(&self) -> PathBuf {
        self.cache_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let mut temp_cache = env::temp_dir();
                temp_cache.push(format!("{}/cache", env!("CARGO_PKG_NAME")));
                temp_cache
            })
    }

    pub fn get_cache_mode(&self) -> CacheMode {
        match &self.cache_mode {
            Some(cache_mode) => {
                match cache_mode.as_str() {
                    "Default" => CacheMode::Default,
                    "ForceCache" => CacheMode::ForceCache,
                    "NoCache" => CacheMode::NoCache,
                    "IgnoreRules" => CacheMode::IgnoreRules,
                    _ => CacheMode::Default, // Fallback
                }
            }
            _ => CacheMode::Default
        }
    }
}
