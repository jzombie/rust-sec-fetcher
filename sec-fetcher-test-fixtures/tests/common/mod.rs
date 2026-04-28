//! Shared fixture-loading helpers for integration tests in this crate.
//!
//! Each integration test file includes this module with `mod common;` at the
//! top of the file.  Only two entry points are exposed:
//!
//! | Function | Returns | Use for |
//! |---|---|---|
//! | [`fixture_string`] | `String` | Raw text fixtures (HTML, XML, `.idx`) |
//! | [`fixture_json`]   | `serde_json::Value` | JSON fixtures (submissions, companyfacts, …) |
//!
//! Every fixture lives on disk as `tests/fixtures/{name}.gz`; the `.gz`
//! suffix is appended automatically by both functions.  Fixtures are created
//! (or refreshed) by running:
//!
//! ```sh
//! cargo run --bin refresh-test-fixtures
//! ```

use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{}.gz", name));
    path
}

/// Decompress `tests/fixtures/{name}.gz` and return the contents as a [`String`].
///
/// Panics with an actionable message if the fixture file is missing.
#[allow(dead_code)]
pub fn fixture_string(name: &str) -> String {
    let path = fixture_path(name);
    let file = File::open(&path).unwrap_or_else(|_| {
        panic!(
            "fixture '{}' not found — run `cargo run --bin refresh-test-fixtures`",
            path.display()
        )
    });
    let mut decoder = GzDecoder::new(file);
    let mut content = String::new();
    decoder
        .read_to_string(&mut content)
        .expect("failed to decompress fixture");
    content
}

/// Decompress `tests/fixtures/{name}.gz` and parse the contents as JSON.
///
/// Panics if the fixture file is missing or its content is not valid JSON.
#[allow(dead_code)]
pub fn fixture_json(name: &str) -> serde_json::Value {
    let path = fixture_path(name);
    let file = File::open(&path).unwrap_or_else(|_| {
        panic!(
            "fixture '{}' not found — run `cargo run --bin refresh-test-fixtures`",
            path.display()
        )
    });
    serde_json::from_reader(GzDecoder::new(file)).expect("fixture is not valid JSON")
}
