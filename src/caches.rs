use simd_r_drive::DataStore;
use std::path::Path;
use std::sync::Arc;

/// Owns the two on-disk `DataStore` instances used by a single `ConfigManager`.
///
/// Each `ConfigManager` creates its own `Caches` from a fresh directory, which
/// means every test that creates a `ConfigManager` gets isolated cache files
/// with no shared state — eliminating cross-test cache pollution via stale
/// entries from previous runs.
pub struct Caches {
    http_cache: Arc<DataStore>,
    preprocessor_cache: Arc<DataStore>,
}

impl std::fmt::Debug for Caches {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Caches")
            .field("http_cache", &"<DataStore>")
            .field("preprocessor_cache", &"<DataStore>")
            .finish()
    }
}

impl Caches {
    /// Opens (or creates) both DataStore files rooted at `base`.
    ///
    /// The directory is created if it does not already exist.
    pub fn open(base: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(base)?;
        let http_cache = Arc::new(
            DataStore::open(&base.join("http_storage_cache.bin"))
                .map_err(|err| format!("Failed to open HTTP DataStore at '{}': {err}", base.display()))?,
        );
        let preprocessor_cache = Arc::new(
            DataStore::open(&base.join("preprocessor_cache.bin"))
                .map_err(|err| format!("Failed to open preprocessor DataStore at '{}': {err}", base.display()))?,
        );
        Ok(Self {
            http_cache,
            preprocessor_cache,
        })
    }

    pub fn get_http_cache_store(&self) -> Arc<DataStore> {
        self.http_cache.clone()
    }

    pub fn get_preprocessor_cache(&self) -> Arc<DataStore> {
        self.preprocessor_cache.clone()
    }
}
