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
            DataStore::open(&base.join("http_storage_cache.bin")).map_err(|err| {
                format!(
                    "Failed to open HTTP DataStore at '{}': {err}",
                    base.display()
                )
            })?,
        );
        let preprocessor_cache = Arc::new(
            DataStore::open(&base.join("preprocessor_cache.bin")).map_err(|err| {
                format!(
                    "Failed to open preprocessor DataStore at '{}': {err}",
                    base.display()
                )
            })?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_caches_open_creates_directory() {
        let temp_dir = std::env::temp_dir().join("caches_test_open");
        let _ = std::fs::remove_dir_all(&temp_dir);

        assert!(!temp_dir.exists());
        let caches = Caches::open(&temp_dir).expect("Caches::open failed");
        assert!(temp_dir.exists(), "Cache directory should have been created");

        let http_cache = caches.get_http_cache_store();
        let preprocessor_cache = caches.get_preprocessor_cache();

        // Both stores should be valid Arc<DataStore> instances
        assert_eq!(
            Arc::as_ptr(&http_cache) as *const (),
            Arc::as_ptr(&caches.get_http_cache_store()) as *const (),
            "get_http_cache_store should return the same Arc"
        );
        assert_eq!(
            Arc::as_ptr(&preprocessor_cache) as *const (),
            Arc::as_ptr(&caches.get_preprocessor_cache()) as *const (),
            "get_preprocessor_cache should return the same Arc"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_caches_open_existing_directory() {
        let temp_dir = std::env::temp_dir().join("caches_test_existing");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let result = Caches::open(&temp_dir);
        assert!(result.is_ok(), "Caches::open on existing dir failed: {:?}", result.err());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_caches_debug_format() {
        let temp_dir = std::env::temp_dir().join("caches_test_debug");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let caches = Caches::open(&temp_dir).unwrap();
        let debug = format!("{:?}", caches);
        assert!(debug.contains("Caches"));
        assert!(debug.contains("http_cache"));
        assert!(debug.contains("preprocessor_cache"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
