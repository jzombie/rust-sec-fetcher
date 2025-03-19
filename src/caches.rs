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
        let cache_path = &config_manager.get_config().get_http_cache_storage_bin(); // Fetch from config

        let data_store = DataStore::open(Path::new(&cache_path))
            .unwrap_or_else(|err| panic!("Failed to open datastore: {}", err));

        SIMD_R_DRIVE_HTTP_CACHE
            .set(Arc::new(data_store))
            .ok()
            .expect("SIMD_R_DRIVE_HTTP_CACHE was already initialized");
    }

    /// Returns a reference to the shared `DataStore`. Panics if not initialized.
    pub fn get_http_cache_store() -> Arc<DataStore> {
        SIMD_R_DRIVE_HTTP_CACHE
            .get()
            .expect("SIMD_R_DRIVE_HTTP_CACHE is uninitialized. Call `Caches::init(config_manager)` first.")
            .clone()
    }
}
