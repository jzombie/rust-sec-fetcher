use crate::config::ConfigManager;
use simd_r_drive::DataStore;
use std::path::Path;
use std::sync::{Arc, OnceLock};

static SIMD_R_DRIVE_HTTP_CACHE: OnceLock<Arc<DataStore>> = OnceLock::new();

pub struct Caches;

impl Caches {
    /// Initializes the shared cache with a dynamic path from `ConfigManager`.
    /// Should be called once before using `get_http_cache()`.
    pub fn init(config_manager: &ConfigManager) {
        // Note: Subsequent calls are effectively no-ops. This is safe to call multiple times (for testing purposes),
        // but they will not reinitialize the cache.
        SIMD_R_DRIVE_HTTP_CACHE.get_or_init(|| {
            let cache_path = &config_manager.get_config().get_http_cache_storage_bin();

            let data_store = DataStore::open(Path::new(cache_path))
                .unwrap_or_else(|err| panic!("Failed to open datastore: {}", err));

            Arc::new(data_store)
        });
    }

    /// Returns a reference to the shared `DataStore`. Panics if not initialized.
    pub fn get_http_cache_store() -> Arc<DataStore> {
        SIMD_R_DRIVE_HTTP_CACHE
            .get()
            .expect("SIMD_R_DRIVE_HTTP_CACHE is uninitialized. Call `Caches::init(config_manager)` first.")
            .clone()
    }
}
