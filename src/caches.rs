use crate::config::ConfigManager;
use log::warn;
use simd_r_drive::DataStore;
use std::path::Path;
use std::sync::{Arc, OnceLock};

static SIMD_R_DRIVE_HTTP_CACHE: OnceLock<Arc<DataStore>> = OnceLock::new();
static SIMD_R_DRIVE_COMPANY_TICKER_CACHE: OnceLock<Arc<DataStore>> = OnceLock::new();

pub struct Caches;

impl Caches {
    /// Initializes the shared cache with a dynamic path from `ConfigManager`.
    /// If already initialized, it logs a warning instead of erroring.
    pub fn init(config_manager: &ConfigManager) {
        // HTTP Cache
        {
            let cache_path = &config_manager.get_config().get_http_cache_storage_bin(); // Fetch from config

            let data_store = DataStore::open(Path::new(&cache_path))
                .unwrap_or_else(|err| panic!("Failed to open datastore: {}", err));

            if SIMD_R_DRIVE_HTTP_CACHE.set(Arc::new(data_store)).is_err() {
                warn!(
                    "SIMD_R_DRIVE_HTTP_CACHE was already initialized. Ignoring reinitialization."
                );
            }
        }
    }

    /// Returns a reference to the shared `DataStore`. Panics if not initialized.
    pub fn get_http_cache_store() -> Arc<DataStore> {
        SIMD_R_DRIVE_HTTP_CACHE
            .get()
            .expect("SIMD_R_DRIVE_HTTP_CACHE is uninitialized. Call `Caches::init(config_manager)` first.")
            .clone()
    }
}
