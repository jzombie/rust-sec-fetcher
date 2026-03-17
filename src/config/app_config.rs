use merge::Merge;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use std::path::PathBuf;

#[cfg(not(debug_assertions))]
use std::sync::OnceLock;

/// Always replace `Some(value)` with `Some(new_value)`
fn overwrite_option<T>(base: &mut Option<T>, new: Option<T>) {
    if let Some(value) = new {
        *base = Some(value);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Merge, JsonSchema)]
#[serde(deny_unknown_fields)] // This ensures unknown keys cause an error
pub struct AppConfig {
    #[merge(strategy = overwrite_option)] // Always replace with new value
    pub email: Option<String>,

    /// Optional override for the app name sent in the User-Agent header.
    /// Defaults to the crate name (`sec-fetcher`) when not set.
    #[merge(strategy = overwrite_option)]
    pub app_name: Option<String>,

    /// Optional override for the app version sent in the User-Agent header.
    /// Defaults to the crate version when not set.
    #[merge(strategy = overwrite_option)]
    pub app_version: Option<String>,

    #[merge(strategy = overwrite_option)]
    pub max_concurrent: Option<usize>,

    #[merge(strategy = overwrite_option)]
    pub min_delay_ms: Option<u64>,

    #[merge(strategy = overwrite_option)]
    pub max_retries: Option<usize>,

    #[merge(strategy = overwrite_option)]
    pub cache_base_dir: Option<PathBuf>,
}

#[cfg(not(debug_assertions))]
static DEFAULT_CACHE_DIR: OnceLock<tempfile::TempDir> = OnceLock::new();

impl Default for AppConfig {
    fn default() -> Self {
        // For testing/example purposes, use a fixed "data" directory in the project root when in debug mode.
        #[cfg(debug_assertions)]
        let cache_base_dir = PathBuf::from("data");

        // simd-r-drive doesn't support safe multi-process access, so each
        // instance gets its own temp dir within the system temp directory.
        #[cfg(not(debug_assertions))]
        let cache_base_dir = DEFAULT_CACHE_DIR
            .get_or_init(|| {
                tempfile::Builder::new()
                    .prefix(env!("CARGO_PKG_NAME"))
                    .tempdir()
                    .expect("failed to create temp cache dir")
            })
            .path()
            .to_path_buf();

        // SEC guidance (Accessing EDGAR Data) currently states a maximum
        // request rate of 10 requests/second. See:
        // https://www.sec.gov/os/accessing-edgar-data
        //
        // This project uses a conservative default to avoid throttling and to
        // be a good citizen: default to 500 ms minimum delay between requests
        // (i.e., ~2 requests/second) with a single concurrent requester.
        Self {
            email: None,
            app_name: None,
            app_version: None,
            max_concurrent: Some(1),
            min_delay_ms: Some(500),
            max_retries: Some(5),
            cache_base_dir: Some(cache_base_dir),
        }
    }
}

impl AppConfig {
    pub fn pretty_print(&self) -> String {
        to_string_pretty(self).unwrap_or_else(|_| "Failed to serialize config".to_string())
    }

    /// Returns a dynamically generated list of valid keys with their types
    pub fn get_valid_keys() -> Vec<(String, String)> {
        let schema = schema_for!(AppConfig);
        schema
            .get("properties")
            .and_then(|p| p.as_object())
            .into_iter()
            .flat_map(|props| {
                props.iter().map(|(key, value)| {
                    let type_name = Self::extract_type_name(value);
                    (key.clone(), type_name)
                })
            })
            .collect()
    }

    /// Extracts the expected type of a field from its schema representation.
    ///
    /// This function is used to determine the expected data type of each field in the
    /// `AppConfig` struct based on its JSON Schema definition. The type information
    /// is extracted from the `Schema` object provided by `schemars`.
    ///
    /// # Arguments
    /// - `schema`: A reference to a `serde_json::Value` describing a field in `AppConfig`.
    ///
    /// # Returns
    /// - A `String` representing the type name(s), joined with ` | ` when multiple.
    /// - `"Unknown"` when type information is unavailable.
    fn extract_type_name(schema: &serde_json::Value) -> String {
        fn title_case(s: &str) -> String {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        }
        // Direct type field — e.g. { "type": "string" } or { "type": ["string", "null"] }
        if let Some(type_val) = schema.get("type") {
            return match type_val {
                serde_json::Value::String(s) => title_case(s),
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(title_case)
                    .collect::<Vec<_>>()
                    .join(" | "),
                _ => "Unknown".to_string(),
            };
        }
        // schemars 1.x emits Option<T> as anyOf: [T-schema, { "type": "null" }]
        if let Some(any_of) = schema.get("anyOf").and_then(|a| a.as_array()) {
            let types: Vec<String> = any_of
                .iter()
                .filter_map(|s| s.get("type")?.as_str().map(title_case))
                .collect();
            if !types.is_empty() {
                return types.join(" | ");
            }
        }
        "Unknown".to_string()
    }

    // Returns the HTTP cache directory path as a `PathBuf` instance.
    //
    // If a HTTP cache directory is explicitly configured, it returns that path.
    // Otherwise, it defaults to a subdirectory within the system's temporary directory,
    // using the package name (from Cargo) to avoid conflicts.
    //
    // # Returns
    // - `PathBuf` pointing to the cache directory.
    // pub fn get_http_cache_storage_bin(&self) -> PathBuf {
    //     self.http_cache_storage_bin
    //         .as_ref()
    //         .map(PathBuf::from)
    //         .unwrap_or_default()
    // }

    // TODO: Remove if using new cache policy
    // TODO: Document
    // https://docs.rs/http-cache/0.20.1/http_cache/enum.CacheMode.html
    // pub fn get_http_cache_mode(&self) -> Result<CacheMode, Box<dyn std::error::Error>> {
    //     let cache_mode = match &self.http_cache_mode {
    //         Some(cache_mode) => match cache_mode.as_str() {
    //             "Default" => CacheMode::Default,
    //             "NoStore" => CacheMode::NoStore,
    //             "Reload" => CacheMode::Reload,
    //             "NoCache" => CacheMode::NoCache,
    //             "ForceCache" => CacheMode::ForceCache,
    //             "OnlyIfCached" => CacheMode::OnlyIfCached,
    //             "IgnoreRules" => CacheMode::IgnoreRules,
    //             _ => return Err(format!("Unhandled cache mode: {}", cache_mode).into()),
    //         },
    //         _ => CacheMode::Default,
    //     };

    //     Ok(cache_mode)
    // }
}
