use merge::Merge;
use schemars::schema::{Schema, SchemaObject, SingleOrVec};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use std::path::PathBuf;

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

    #[merge(strategy = overwrite_option)]
    pub max_concurrent: Option<usize>,

    #[merge(strategy = overwrite_option)]
    pub min_delay_ms: Option<u64>,

    #[merge(strategy = overwrite_option)]
    pub max_retries: Option<usize>,

    #[merge(strategy = overwrite_option)]
    pub http_cache_storage_bin: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        let temp_cache_dir = {
            // TODO: Use temp
            // let mut temp_cache = env::temp_dir();
            // temp_cache.push(format!("{}/http_storage_cache.bin", env!("CARGO_PKG_NAME")));
            // temp_cache

            // TODO: Remove
            PathBuf::from("data/http_storage_cache.bin")
        };

        Self {
            email: None,
            max_concurrent: Some(1),
            min_delay_ms: Some(1000),
            max_retries: Some(5),
            http_cache_storage_bin: Some(temp_cache_dir.to_string_lossy().into_owned()),
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
            .schema
            .object
            .as_ref()
            .unwrap()
            .properties
            .iter()
            .map(|(key, value)| {
                let type_name = Self::extract_type_name(value);
                (key.clone(), type_name)
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
    /// - `schema`: A reference to a `Schema` object describing a field in `AppConfig`.
    ///
    /// # Returns
    /// - A `String` representing the type name. If the schema specifies multiple possible
    ///   types, they are joined with `|` (e.g., `"String | Null"`).
    /// - If the type information is unavailable, `"Unknown"` is returned.
    ///
    /// # Example
    /// Given the following schema for a field:
    /// ```json
    /// {
    ///   "type": ["string", "null"]
    /// }
    /// ```
    /// The function would return:
    /// ```
    /// "String | Null"
    /// ```
    fn extract_type_name(schema: &Schema) -> String {
        if let Schema::Object(SchemaObject {
            instance_type: Some(types),
            ..
        }) = schema
        {
            match types {
                SingleOrVec::Single(t) => format!("{:?}", t), // Single type
                SingleOrVec::Vec(vec) => vec
                    .iter()
                    .map(|t| format!("{:?}", t))
                    .collect::<Vec<_>>()
                    .join(" | "), // Multiple types
            }
        } else {
            "Unknown".to_string()
        }
    }

    /// Returns the HTTP cache directory path as a `PathBuf` instance.
    ///
    /// If a HTTP cache directory is explicitly configured, it returns that path.
    /// Otherwise, it defaults to a subdirectory within the system's temporary directory,
    /// using the package name (from Cargo) to avoid conflicts.
    ///
    /// # Returns
    /// - `PathBuf` pointing to the cache directory.
    pub fn get_http_cache_storage_bin(&self) -> PathBuf {
        self.http_cache_storage_bin
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_default()
    }

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
