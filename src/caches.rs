use crate::config::ConfigManager;
use log::warn;
use simd_r_drive::DataStore;
use std::path::Path;
use std::sync::{Arc, OnceLock};

static SIMD_R_DRIVE_HTTP_CACHE: OnceLock<Arc<DataStore>> = OnceLock::new();
static SIMD_R_DRIVE_PREPROCESSOR_CACHE: OnceLock<Arc<DataStore>> = OnceLock::new();

pub struct Caches;

impl Caches {
    /// Initializes the shared cache with a dynamic path from `ConfigManager`.
    /// If already initialized, it logs a warning instead of erroring.
    pub fn init(config_manager: &ConfigManager) {
        let cache_base_path = config_manager.get_config().cache_base_dir.as_ref().unwrap(); // Fetch from config

        // HTTP Cache
        {
            let http_cache_path = cache_base_path.join("http_storage_cache.bin");

            let data_store = DataStore::open(Path::new(&http_cache_path))
                .unwrap_or_else(|err| panic!("Failed to open datastore: {}", err));

            if SIMD_R_DRIVE_HTTP_CACHE.set(Arc::new(data_store)).is_err() {
                warn!(
                    "SIMD_R_DRIVE_HTTP_CACHE was already initialized. Ignoring reinitialization."
                );
            }
        }

        // Company ticker Cache
        {
            let http_cache_path = cache_base_path.join("preprocessor_cache.bin");

            let data_store = DataStore::open(Path::new(&http_cache_path))
                .unwrap_or_else(|err| panic!("Failed to open datastore: {}", err));

            if SIMD_R_DRIVE_PREPROCESSOR_CACHE
                .set(Arc::new(data_store))
                .is_err()
            {
                warn!(
                    "SIMD_R_DRIVE_PREPROCESSOR_CACHE was already initialized. Ignoring reinitialization."
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

    /// Returns a reference to the shared `DataStore`. Panics if not initialized.
    pub fn get_preprocessor_cache() -> Arc<DataStore> {
        SIMD_R_DRIVE_PREPROCESSOR_CACHE
            .get()
            .expect("SIMD_R_DRIVE_PREPROCESSOR_CACHE is uninitialized. Call `Caches::init(config_manager)` first.")
            .clone()
    }
}
