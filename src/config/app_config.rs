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
    pub cache_dir: Option<String>,
    pub cache_mode: Option<String>,
}

// TODO: Implement default config

impl AppConfig {
    pub fn pretty_print(&self) -> String {
        to_string_pretty(self).unwrap_or_else(|_| "Failed to serialize config".to_string())
    }

    pub fn get_cache_dir(&self) -> Option<PathBuf> {
        self.cache_dir.clone().map(|dir_string| PathBuf::from(dir_string))
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
