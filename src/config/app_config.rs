use std::env;
use serde::{Serialize, Deserialize};
use serde_json::to_string_pretty;
use std::path::PathBuf;
use http_cache_reqwest::{Cache, CacheMode, CACacheManager, HttpCache, HttpCacheOptions};
use merge::Merge;

#[derive(Debug, Serialize, Deserialize, Clone, Merge)]
pub struct AppConfig {
    pub email: Option<String>,
    pub max_concurrent: Option<usize>,
    pub min_delay_ms: Option<u64>,
    pub max_retries: Option<usize>,
    pub http_cache_dir: Option<String>,
    pub http_cache_mode: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        let temp_cache_dir = {
            let mut temp_cache = env::temp_dir();
            temp_cache.push(format!("{}/cache", env!("CARGO_PKG_NAME")));
            temp_cache
        };

        Self {
            email: None,
            max_concurrent: Some(1),
            min_delay_ms: Some(1000),
            max_retries: Some(5),
            http_cache_dir: Some(temp_cache_dir.to_string_lossy().into_owned()),
            http_cache_mode: None
        }
    }
}

impl AppConfig {
    pub fn pretty_print(&self) -> String {
        to_string_pretty(self).unwrap_or_else(|_| "Failed to serialize config".to_string())
    }

    /// Returns the HTTP cache directory path as a `PathBuf` instance.
    ///
    /// If a HTTP cache directory is explicitly configured, it returns that path.
    /// Otherwise, it defaults to a subdirectory within the system's temporary directory,
    /// using the package name (from Cargo) to avoid conflicts.
    ///
    /// # Returns
    /// - `PathBuf` pointing to the cache directory.
    pub fn get_http_cache_dir(&self) -> PathBuf {
        self.http_cache_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_default()
    }

    // TODO: Document
    // https://docs.rs/http-cache/0.20.1/http_cache/enum.CacheMode.html
    pub fn get_http_cache_mode(&self) -> Result<CacheMode, Box<dyn std::error::Error>> {
        let cache_mode = match &self.http_cache_mode {
            Some(cache_mode) => {
                match cache_mode.as_str() {
                    "Default" => CacheMode::Default,
                    "NoStore" => CacheMode::NoStore,
                    "Reload" => CacheMode::Reload,
                    "NoCache" => CacheMode::NoCache,
                    "ForceCache" => CacheMode::ForceCache,
                    "OnlyIfCached" => CacheMode::OnlyIfCached,
                    "IgnoreRules" => CacheMode::IgnoreRules,
                    _ => return Err(format!("Unhandled cache mode: {}", cache_mode).into())
                }
            }
            _ => CacheMode::Default
        };

        Ok(cache_mode)
    }
}
