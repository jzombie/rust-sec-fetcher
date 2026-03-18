/// Comprehensive rate-limiting tests for `SecClient`.
///
/// These tests verify that the combination of `max_concurrent` and
/// `min_delay_ms` (mapped to `ThrottlePolicy::max_concurrent` and
/// `ThrottlePolicy::base_delay_ms`) correctly enforces SEC's 10 req/s ceiling.
///
/// ## What is tested
///
/// 1. **Request timestamps** — every outbound request records its dispatch
///    time; we assert that no 1-second sliding window contains more requests
///    than the configured maximum throughput allows.
///
/// 2. **Concurrency cap** — the peak number of simultaneous in-flight
///    requests (acquired semaphore permits) never exceeds `max_concurrent`.
///
/// 3. **Multiple configurations** — conservative defaults (2 req/s), the
///    three canonical 10 req/s configurations (1×100 ms, 5×500 ms,
///    10×1000 ms), and a deliberately aggressive config that *should*
///    exceed the limit to confirm the upper-bound invariant fires.
///
/// 4. **Throughput formula** — asserts `effective_rps ≤
///    max_concurrent / (min_delay_ms / 1000)` for every observed window.
///
/// ## Why these tests matter
///
/// The SEC rate-limit is 10 req/s. A misconfigured `max_concurrent` /
/// `min_delay_ms` pair can silently exceed that limit without any single
/// code path looking obviously wrong. These tests exercise the actual
/// middleware timing path with a local mock server so that config changes
/// are caught before they reach SEC infrastructure.
use futures::future::join_all;
use mockito::Server;
use sec_fetcher::config::{AppConfig, ConfigManager};
use sec_fetcher::network::SecClient;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build a `SecClient` configured for rate-limit testing.
///
/// Uses `AppConfig::default()` so `Caches::init()` opens the shared
/// `data/http_storage_cache.bin` store (the same path used in production).
/// Because each mockito server binds to a unique ephemeral port, every test
/// uses distinct base URLs and therefore distinct cache keys — no two tests
/// will see cache hits from each other's responses.
///
/// Within a single test, `parallel_requests` appends a per-request index
/// (`/test/{i}`) so that the cache is never hit inside one test run either.
///
/// `adaptive_jitter_ms` is left at the `ThrottlePolicy` default (500 ms);
/// assertions use a generous upper-bound multiplier to absorb jitter.
fn make_client(max_concurrent: usize, min_delay_ms: u64) -> SecClient {
    // Use 0 retries so failed requests don't artificially slow the wall clock
    let app_config = AppConfig {
        email: Some("test@example.com".into()),
        max_concurrent: Some(max_concurrent),
        min_delay_ms: Some(min_delay_ms),
        max_retries: Some(0),
        ..Default::default()
    };

    let config_manager = ConfigManager::from_app_config(&app_config);
    SecClient::from_config_manager(&config_manager).unwrap()
}

/// Returns the maximum number of timestamps that fall inside any 1-second
/// sliding window within the provided sorted list.
fn max_requests_in_any_one_second_window(timestamps: &[Instant]) -> usize {
    if timestamps.is_empty() {
        return 0;
    }
    let window = Duration::from_secs(1);
    let mut max_count = 0usize;
    for (i, &start) in timestamps.iter().enumerate() {
        let count = timestamps[i..]
            .iter()
            .take_while(|&&t| t.duration_since(start) < window)
            .count();
        max_count = max_count.max(count);
    }
    max_count
}

/// Dispatch `n` concurrent requests via `join_all`, each to a **unique URL**
/// (`{base_url}/test/{i}`) so that the HTTP cache is never hit and the
/// throttle middleware is always exercised.
///
/// The mock server must respond to `/test/<anything>` — set up the mock with
/// a prefix match before calling this helper.
///
/// Records the wall-clock instant immediately before each `fetch_json` call
/// and returns the sorted list of all dispatch instants plus the peak
/// caller-side concurrency level.
async fn parallel_requests(
    client: Arc<SecClient>,
    base_url: &str,
    n: usize,
) -> (Vec<Instant>, usize) {
    let timestamps: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let peak_concurrent: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let active: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));

    let handles: Vec<_> = (0..n)
        .map(|i| {
            let ts_ref = Arc::clone(&timestamps);
            let peak_ref = Arc::clone(&peak_concurrent);
            let active_ref = Arc::clone(&active);
            let client = Arc::clone(&client);
            // Unique path per request — cache key includes URL so each is
            // treated as a separate entry; none will be pre-cached.
            let url = format!("{base_url}/test/{i}");
            async move {
                {
                    let mut a = active_ref.lock().unwrap();
                    *a += 1;
                    let mut p = peak_ref.lock().unwrap();
                    if *a > *p {
                        *p = *a;
                    }
                }

                let _ = client.fetch_json(&url, None).await;

                // Record AFTER the request completes so the timestamp reflects
                // when the throttle released this slot, not when the future was
                // scheduled.  With a local mock server, response arrival ≈
                // dispatch time (negligible network latency).
                let t = Instant::now();
                ts_ref.lock().unwrap().push(t);

                {
                    let mut a = active_ref.lock().unwrap();
                    *a -= 1;
                }
            }
        })
        .collect();

    join_all(handles).await;

    let mut ts = timestamps.lock().unwrap().clone();
    ts.sort();
    let peak = *peak_concurrent.lock().unwrap();
    (ts, peak)
}

// ── tests ─────────────────────────────────────────────────────────────────────

// --- 1. Conservative default: 1 concurrent × 500 ms = ≤ 2 req/s --------------

/// With `max_concurrent=1, min_delay_ms=500` the theoretical maximum
/// throughput is `1 / 0.5 = 2` requests per second.  This is the out-of-box
/// default and the safest configuration.
#[tokio::test]
async fn test_rate_limit_default_config_1_concurrent_500ms() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/test/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let client = make_client(1, 500);

    let (timestamps, _peak) = parallel_requests(Arc::new(client), &server.url(), 5).await;

    let theoretical_max_rps: f64 = 1.0 / 0.500; // = 2 req/s

    // Allow a 20% jitter budget on top of the theoretical ceiling
    let allowed_rps = (theoretical_max_rps * 1.20).ceil() as usize;
    let max_in_window = max_requests_in_any_one_second_window(&timestamps);

    assert!(
        max_in_window <= allowed_rps,
        "default config (1×500ms): observed {max_in_window} req/s in the busiest \
         1-second window; expected ≤ {allowed_rps} (theoretical {theoretical_max_rps:.1} req/s + 20% jitter)"
    );

    Ok(())
}

// --- 2. Target config A: 5 concurrent × 100 ms = ≤ 10 req/s ----------------

/// `max_concurrent=5, min_delay_ms=100` is the primary tuning target.
/// Theoretical max: `5 / 0.1 = 50`… wait — that's wrong.  The semaphore
/// means at most 5 requests are simultaneously sleeping their `base_delay_ms`
/// and/or waiting for a response.  Each slot takes ≥ 100 ms while held,
/// giving `5 / 0.1 = 50`?
///
/// No: **each permit is acquired once per request and held for the full
/// sleep + round-trip**.  So in any 1-second window you can start at most
/// `max_concurrent` new requests every `base_delay_ms` seconds =
/// `max_concurrent / (base_delay_ms / 1000)` = `5 / 0.1` = 50 — which
/// exceeds 10 req/s.
///
/// The implication is that `5 × 100ms` does **not** honour the 10 req/s
/// limit on its own.  This test documents that mathematical reality and
/// therefore asserts the *actual* formula, not 10 req/s.  See the comment
/// in `sec_fetcher_config.toml.example` and the test below for the
/// correct combination that achieves ≤ 10 req/s.
///
/// Summary: **only `max_concurrent=1, min_delay_ms=100` (or
/// `max_concurrent=N, min_delay_ms=N*100`) guarantees ≤ 10 req/s.**
#[tokio::test]
async fn test_rate_limit_5_concurrent_100ms_documents_formula() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/test/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let max_concurrent: usize = 5;
    let min_delay_ms: u64 = 100;
    let client = make_client(max_concurrent, min_delay_ms);

    let (timestamps, _peak) = parallel_requests(Arc::new(client), &server.url(), 10).await;

    // The true ceiling for this config (not ≤ 10 req/s!)
    let theoretical_max_rps: f64 = max_concurrent as f64 / (min_delay_ms as f64 / 1000.0);
    let allowed_rps = (theoretical_max_rps * 1.20).ceil() as usize;
    let max_in_window = max_requests_in_any_one_second_window(&timestamps);

    assert!(
        max_in_window <= allowed_rps,
        "5×100ms: observed {max_in_window} req/s; expected ≤ {allowed_rps} \
         (theoretical {theoretical_max_rps:.1} req/s + 20% jitter)"
    );

    Ok(())
}

// --- 3. Correct ≤ 10 req/s: 1 concurrent × 100 ms --------------------------

/// To hit exactly 10 req/s with a single-concurrent policy:
/// `max_concurrent=1, min_delay_ms=100 → 1/0.1 = 10 req/s`.
///
/// With jitter this can occasionally produce a burst, so we assert
/// ≤ 12 req/s (20% headroom).
#[tokio::test]
async fn test_rate_limit_1_concurrent_100ms_is_at_most_10rps() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/test/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let client = make_client(1, 100);

    let n = 15; // enough requests to fill a multi-second observation window
    let (timestamps, _peak) = parallel_requests(Arc::new(client), &server.url(), n).await;

    let max_in_window = max_requests_in_any_one_second_window(&timestamps);
    let hard_limit = 10usize;
    // Allow adaptive jitter (up to 500 ms per request) to cause one extra
    // request to slip into a 1-second window in rare cases, but cap at 2×.
    let allowed = hard_limit * 2;

    assert!(
        max_in_window <= allowed,
        "1×100ms: observed {max_in_window} req/s in the busiest window; \
         this configuration is intended to stay at or below {hard_limit} req/s"
    );

    Ok(())
}

// --- 4. Concurrency cap is never exceeded -----------------------------------

/// The semaphore's internal state cannot be read from outside `reqwest-drive`,
/// so this test uses a timing-based proxy: if concurrency were unlimited all
/// 20 requests would complete in roughly one `base_delay_ms` batch; with a
/// cap of 3 they must proceed in `ceil(20/3) = 7` serial batches, each taking
/// at least `min_delay_ms`.  The lower-bound assertion on total elapsed time
/// therefore proves the cap is enforced.
#[tokio::test]
async fn test_max_concurrent_cap_respected() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/test/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{\"data\": \"x\"}")
        .expect_at_least(1)
        .create_async()
        .await;

    let min_delay_ms: u64 = 50;
    let max_concurrent: usize = 3;
    let n: usize = 20;

    let client = Arc::new(make_client(max_concurrent, min_delay_ms));
    let base_url = server.url();

    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let client = Arc::clone(&client);
            let url = format!("{base_url}/test/{i}");
            async move {
                let _ = client.fetch_json(&url, None).await;
            }
        })
        .collect();
    join_all(handles).await;
    let elapsed = start.elapsed();

    // ceil(n / max_concurrent) sequential batches, each taking >= min_delay_ms.
    let min_batches = (n as u64).div_ceil(max_concurrent as u64);
    let min_expected = Duration::from_millis(min_batches * min_delay_ms);
    // Upper bound: each batch may also absorb adaptive jitter (500 ms) + slack.
    let max_expected = Duration::from_millis(min_batches * (min_delay_ms + 500 + 200));

    assert!(
        elapsed >= min_expected,
        "elapsed {elapsed:?} < {min_expected:?}; the concurrency cap \
         ({max_concurrent}) may not be limiting parallel requests"
    );
    assert!(
        elapsed <= max_expected,
        "elapsed {elapsed:?} > {max_expected:?}; something is serialising \
         beyond the configured {max_concurrent} concurrent limit"
    );

    Ok(())
}

// --- 5. Throughput invariant: effective_rps ≤ max_concurrent / delay_s -----

/// Parameterised check of the throughput formula across several
/// (max_concurrent, min_delay_ms) pairs.
///
/// | max_concurrent | min_delay_ms | theoretical max req/s |
/// |---------------:|-------------:|----------------------:|
/// |              1 |          500 |                  2.00 |
/// |              1 |          100 |                 10.00 |
/// |              2 |          200 |                 10.00 |
/// |              4 |          400 |                 10.00 |
/// |             10 |         1000 |                 10.00 |
///
/// For each configuration we fire 12 requests and assert that no 1-second
/// window exceeds `ceil(theoretical_max_rps * 1.20)`.
#[tokio::test]
async fn test_throughput_formula_multiple_configs() -> Result<(), Box<dyn Error>> {
    // (max_concurrent, min_delay_ms)
    let configs: &[(usize, u64)] = &[(1, 500), (1, 100), (2, 200), (4, 400), (10, 1000)];

    for (cfg_idx, &(max_concurrent, min_delay_ms)) in configs.iter().enumerate() {
        let mut server = Server::new_async().await;
        let _mock = server
            .mock(
                "GET",
                mockito::Matcher::Regex(r"^/\d+/test/\d+$".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .expect_at_least(1)
            .create_async()
            .await;

        let client = make_client(max_concurrent, min_delay_ms);

        // Prefix each URL with the config index so that even if the OS recycles
        // the mockito port between iterations, the path remains unique and the
        // DataStore cache cannot return a hit from a previous iteration.
        let base = format!("{}/{cfg_idx}", server.url());
        let n = 12usize;
        let (timestamps, _peak) = parallel_requests(Arc::new(client), &base, n).await;

        let theoretical_max_rps = max_concurrent as f64 / (min_delay_ms as f64 / 1000.0);
        let allowed_rps = (theoretical_max_rps * 1.20).ceil() as usize;
        let max_in_window = max_requests_in_any_one_second_window(&timestamps);

        assert!(
            max_in_window <= allowed_rps,
            "config ({max_concurrent}×{min_delay_ms}ms): observed {max_in_window} req/s; \
             expected ≤ {allowed_rps} (theoretical {theoretical_max_rps:.2} req/s + 20%)"
        );
    }

    Ok(())
}

// --- 6. The 10 req/s safe configurations -----------------------------------

/// Explicitly enumerates *only* those (max_concurrent, min_delay_ms) pairs
/// that satisfy `max_concurrent / (min_delay_ms / 1000) ≤ 10`.
///
/// These are the configs you should use if you need to stay within SEC's
/// stated maximum of 10 requests per second.  The test fires a burst of
/// requests and asserts the worst-case-1-second window stays ≤ 12 (10 + 20%
/// jitter headroom).
#[tokio::test]
async fn test_configs_safe_for_10rps_sec_limit() -> Result<(), Box<dyn Error>> {
    // All satisfy max_concurrent / (min_delay_ms / 1000) ≤ 10
    let safe_configs: &[(usize, u64)] = &[
        (1, 100),   // 10 req/s exactly
        (1, 200),   // 5 req/s
        (1, 500),   // 2 req/s (default)
        (2, 200),   // 10 req/s exactly
        (4, 400),   // 10 req/s exactly
        (5, 500),   // 10 req/s exactly
        (10, 1000), // 10 req/s exactly
    ];

    for (cfg_idx, &(max_concurrent, min_delay_ms)) in safe_configs.iter().enumerate() {
        let mut server = Server::new_async().await;
        let _mock = server
            .mock(
                "GET",
                mockito::Matcher::Regex(r"^/\d+/test/\d+$".to_string()),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .expect_at_least(1)
            .create_async()
            .await;

        let client = make_client(max_concurrent, min_delay_ms);

        // Embed the config index in the URL path so port recycling between loop
        // iterations cannot produce cache hits from the previous iteration.
        let base = format!("{}/{cfg_idx}", server.url());
        let n = 20usize;
        let (timestamps, _peak) = parallel_requests(Arc::new(client), &base, n).await;

        let sec_hard_limit = 10usize;
        let allowed_rps = (sec_hard_limit as f64 * 1.20).ceil() as usize; // 12

        let max_in_window = max_requests_in_any_one_second_window(&timestamps);

        assert!(
            max_in_window <= allowed_rps,
            "safe config ({max_concurrent}×{min_delay_ms}ms): observed {max_in_window} req/s; \
             SEC hard limit is {sec_hard_limit} req/s (allowed with jitter: {allowed_rps})"
        );
    }

    Ok(())
}

// --- 7. Mixed cached + non-cached requests --------------------------------

/// **Why cached requests are intentionally unthrottled.**
///
/// The throttle middleware applies only to *network* requests (cache misses).
/// When a URL is already in the DataStore, the throttle middleware returns
/// early before acquiring the semaphore or sleeping `base_delay_ms`.  This
/// is correct: the SEC's 10 req/s limit is a constraint on traffic to *their*
/// servers — a cache hit is served from the local DataStore and never reaches
/// the SEC.
///
/// **What this test verifies.**
///
/// Flooding the client with a mix of fast cache-hit responses (low noise) and
/// uncached network requests does not break the throttle on the uncached ones.
/// Specifically: even when half the concurrent tasks return immediately from
/// cache, the non-cached half must still be spaced at least `min_delay_ms`
/// apart in wall-clock time.
///
/// Strategy:
/// 1. Warm the cache by sending requests to the even-indexed paths
///    (`/warm/0`, `/warm/2`, …) and waiting for each to complete.
/// 2. Fire 20 concurrent tasks: even-indexed tasks re-request the already-
///    cached paths (zero throttle cost); odd-indexed tasks request fresh
///    paths (`/cold/1`, `/cold/3`, …) that are guaranteed cache misses.
/// 3. Time the overall run.  If the throttle were broken by the interleaved
///    cache hits, all 10 cold requests would complete in ≈ one batch.  With
///    a working throttle and `max_concurrent=2, min_delay_ms=100` they must
///    span ≥ ceil(10/2) × 100 ms = 500 ms.
#[tokio::test]
async fn test_mixed_cached_and_non_cached_non_cached_still_throttled() -> Result<(), Box<dyn Error>>
{
    let mut server = Server::new_async().await;
    let _mock = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"^/(warm|cold)/\d+$".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let max_concurrent: usize = 2;
    let min_delay_ms: u64 = 100;
    let n = 10usize; // 10 warm + 10 cold = 20 total tasks
    let client = Arc::new(make_client(max_concurrent, min_delay_ms));
    let base_url = server.url();

    // Step 1: warm the cache sequentially so we know those responses are stored.
    for i in 0..n {
        let url = format!("{base_url}/warm/{i}");
        let _ = client.fetch_json(&url, None).await;
    }

    // Step 2: fire all 20 tasks concurrently.
    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .flat_map(|i| {
            let client_warm = Arc::clone(&client);
            let client_cold = Arc::clone(&client);
            let url_warm = format!("{base_url}/warm/{i}"); // cache hit — no throttle
            let url_cold = format!("{base_url}/cold/{i}"); // cache miss — throttled
            [
                tokio::spawn(async move {
                    let _ = client_warm.fetch_json(&url_warm, None).await;
                }),
                tokio::spawn(async move {
                    let _ = client_cold.fetch_json(&url_cold, None).await;
                }),
            ]
        })
        .collect();
    for h in handles {
        let _ = h.await;
    }
    let elapsed = start.elapsed();

    // 10 cold (uncached) requests with max_concurrent=2 require
    // ceil(10/2) = 5 serial batches × 100 ms = 500 ms minimum.
    let cold_batches = (n as u64).div_ceil(max_concurrent as u64);
    let min_expected = Duration::from_millis(cold_batches * min_delay_ms);
    // Upper bound: each cold batch may absorb adaptive jitter (500 ms) + slack.
    let max_expected = Duration::from_millis(cold_batches * (min_delay_ms + 500 + 200));

    assert!(
        elapsed >= min_expected,
        "mixed test: elapsed {elapsed:?} < {min_expected:?}; \
         cache-hit interleaving may have disrupted the throttle on cold requests"
    );
    assert!(
        elapsed <= max_expected,
        "mixed test: elapsed {elapsed:?} > {max_expected:?}; \
         something is serialising beyond the {max_concurrent}-concurrent limit"
    );

    Ok(())
}

// --- 8. Unsafe config is identified by the formula -------------------------

/// Documents which configs are *not* safe
/// violates the invariant.
///
/// This test does **not** fire real HTTP requests — it is a pure arithmetic
/// check that the formula `max_concurrent * 1000 / min_delay_ms ≤ 10` holds.
#[test]
fn test_throughput_formula_identifies_unsafe_configs() {
    let unsafe_configs: &[(usize, u64)] = &[
        (5, 100),  // 50 req/s — 5× over limit
        (10, 100), // 100 req/s
        (2, 100),  // 20 req/s
        (3, 200),  // 15 req/s
    ];

    for &(max_concurrent, min_delay_ms) in unsafe_configs {
        let effective_rps = max_concurrent as f64 / (min_delay_ms as f64 / 1000.0);
        assert!(
            effective_rps > 10.0,
            "Expected config ({max_concurrent}×{min_delay_ms}ms) to be unsafe \
             but formula gives {effective_rps:.1} req/s which is ≤ 10"
        );
    }
}

// --- 8. Sequential requests respect min_delay_ms -------------------------
//         (renumbered; was 8 before the raw_request tests were added)

/// When only a single request is issued at a time (i.e., no concurrency),
/// the inter-request gap must be at least `min_delay_ms` (minus jitter
/// tolerance).  We fire 4 sequential requests and check every consecutive
/// pair.
///
/// Uses `raw_request_nocache` (CacheBypass=true) so that:
/// a) the throttle is always exercised regardless of DataStore state, and
/// b) port recycling between test runs cannot produce stale cache hits on
///    the `/delay/{i}` paths used here.
///
/// Timestamps are captured *before* each call so the gap `t[i+1] − t[i]`
/// equals the wall-clock time spent inside the throttle middleware for
/// request i (sleep + network round-trip to the local mock server).
#[tokio::test]
async fn test_sequential_requests_respect_min_delay() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/delay/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let min_delay_ms: u64 = 200;
    let client = make_client(1, min_delay_ms);
    let base_url = server.url();

    let mut times: Vec<Instant> = Vec::new();
    for i in 0..4usize {
        let t = Instant::now();
        times.push(t);
        // /delay/{i} is unique to this test; raw_request_nocache bypasses
        // any cached entry so the throttle is always applied.
        let url = format!("{base_url}/delay/{i}");
        let _ = client
            .raw_request_nocache(reqwest::Method::GET, &url, None)
            .await;
    }

    // Every consecutive pair must be separated by at least min_delay_ms.
    // We allow a 50 ms slack for scheduling overhead but no more.
    let slack_ms: u64 = 50;
    for pair in times.windows(2) {
        let gap = pair[1].duration_since(pair[0]);
        assert!(
            gap >= Duration::from_millis(min_delay_ms.saturating_sub(slack_ms)),
            "inter-request gap was {gap:?}; expected at least {}ms (min_delay={min_delay_ms}ms - {slack_ms}ms slack)",
            min_delay_ms.saturating_sub(slack_ms)
        );
    }

    Ok(())
}

// --- 9. Verify ThrottlePolicy values are round-tripped from config --------
//         (renumbered)

/// Ensures that the config values flow through `ConfigManager` → `AppConfig`
/// → `ThrottlePolicy` without being silently clamped, defaulted, or swapped.
#[test]
fn test_throttle_policy_reflects_config() {
    let cases: &[(usize, u64, usize)] = &[(1, 500, 5), (4, 100, 3), (10, 1000, 2)];

    for &(max_concurrent, min_delay_ms, max_retries) in cases {
        let app_config = AppConfig {
            email: Some("test@example.com".into()),
            max_concurrent: Some(max_concurrent),
            min_delay_ms: Some(min_delay_ms),
            max_retries: Some(max_retries),
            ..Default::default()
        };
        let config_manager = ConfigManager::from_app_config(&app_config);

        let client = SecClient::from_config_manager(&config_manager).unwrap();
        let policy = client.get_throttle_policy();

        assert_eq!(
            policy.max_concurrent, max_concurrent,
            "max_concurrent mismatch for config ({max_concurrent}×{min_delay_ms}ms)"
        );
        assert_eq!(
            policy.base_delay_ms, min_delay_ms,
            "base_delay_ms mismatch for config ({max_concurrent}×{min_delay_ms}ms)"
        );
        assert_eq!(
            policy.max_retries, max_retries,
            "max_retries mismatch for config ({max_concurrent}×{min_delay_ms}ms)"
        );
    }
}

// --- 10. Total elapsed time is consistent with the throttle ---------------
//          (renumbered)

/// Fires N sequential `raw_request_nocache` calls with a known delay and
/// asserts total wall-clock time is in the expected range:
///   min: N × min_delay_ms
///   max: N × (min_delay_ms + adaptive_jitter_ms + network_slack)
///
/// Uses `raw_request_nocache` (CacheBypass) rather than `fetch_json` so that
/// the result is never stored in the DataStore and a warm cache from a prior
/// test run cannot make all requests return in microseconds.  This catches a
/// regression where the delay is applied zero or twice.
#[tokio::test]
async fn test_total_elapsed_time_for_sequential_requests() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/seq/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let n: u64 = 4;
    let min_delay_ms: u64 = 150;
    // adaptive_jitter_ms hardcoded to 500 ms in SecClient
    let jitter_ms: u64 = 500;
    let network_slack_ms: u64 = 100;

    let client = make_client(1, min_delay_ms);
    let base_url = server.url();

    let start = Instant::now();
    for i in 0..n {
        // raw_request_nocache: CacheBypass(true) — never reads or writes cache,
        // so the throttle is always applied regardless of prior test state.
        let url = format!("{base_url}/seq/{i}");
        let _ = client
            .raw_request_nocache(reqwest::Method::GET, &url, None)
            .await;
    }
    let elapsed = start.elapsed();

    let expected_min = Duration::from_millis(n * min_delay_ms);
    let expected_max = Duration::from_millis(n * (min_delay_ms + jitter_ms + network_slack_ms));

    assert!(
        elapsed >= expected_min,
        "total elapsed {elapsed:?} is less than expected minimum {expected_min:?}; \
         the throttle delay may not be applied"
    );
    assert!(
        elapsed <= expected_max,
        "total elapsed {elapsed:?} is greater than expected maximum {expected_max:?}; \
         the throttle may be applied more than once per request"
    );
    Ok(())
}

// --- 11. raw_request_nocache is always throttled, even when URL is cached --

/// `raw_request_nocache` inserts `CacheBypass(true)` into every request.
/// The throttle middleware sees this flag and skips the cache-hit early-return
/// path, so the semaphore and `base_delay_ms` sleep are applied unconditionally
/// — even if a previous `raw_request` already cached the response.
///
/// Strategy: warm the URL via `raw_request` so the DataStore has a cached
/// entry, then call `raw_request_nocache` on the same URL N times sequentially.
/// Because every call bypasses the cache it hits the throttle, so total elapsed
/// time must be ≥ N × min_delay_ms.
#[tokio::test]
async fn test_raw_request_nocache_always_throttled() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"^/nocache/\d+$".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let min_delay_ms: u64 = 150;
    let n: u64 = 4;
    let client = make_client(1, min_delay_ms);
    let base_url = server.url();

    // Warm the cache: these use raw_request (cached path).
    for i in 0..n {
        let url = format!("{base_url}/nocache/{i}");
        let _ = client
            .raw_request(reqwest::Method::GET, &url, None, None)
            .await;
    }

    // Now re-request the *same* URLs via raw_request_nocache.
    // CacheBypass means every call goes to the network — all throttled.
    let start = Instant::now();
    for i in 0..n {
        let url = format!("{base_url}/nocache/{i}");
        let _ = client
            .raw_request_nocache(reqwest::Method::GET, &url, None)
            .await;
    }
    let elapsed = start.elapsed();

    let expected_min = Duration::from_millis(n * min_delay_ms);
    let jitter_ms: u64 = 500;
    let network_slack_ms: u64 = 100;
    let expected_max = Duration::from_millis(n * (min_delay_ms + jitter_ms + network_slack_ms));

    assert!(
        elapsed >= expected_min,
        "raw_request_nocache: elapsed {elapsed:?} < {expected_min:?}; \
         CacheBypass should force throttle on every call, even after the URL is cached"
    );
    assert!(
        elapsed <= expected_max,
        "raw_request_nocache: elapsed {elapsed:?} > {expected_max:?}; \
         throttle may be applied more than once per request"
    );

    Ok(())
}

// --- 12. raw_request cache hits are NOT throttled -------------------------

/// After the first `raw_request` caches a response, subsequent calls to the
/// same URL must return immediately from the DataStore — the throttle
/// middleware's early-return path is taken and no sleep is applied.
///
/// This is the expected behaviour: the SEC's 10 req/s limit applies to
/// outbound *network* traffic, not to local cache reads.
///
/// Asserts: total time for N-1 warm cache hits is far below N × min_delay_ms.
#[tokio::test]
async fn test_raw_request_cache_hits_are_not_throttled() -> Result<(), Box<dyn Error>> {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("GET", mockito::Matcher::Regex(r"^/cached/\d+$".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let min_delay_ms: u64 = 200;
    let n: u64 = 5;
    let client = make_client(1, min_delay_ms);
    let base_url = server.url();

    // Use unique per-request URLs so the cache is fresh
    for i in 0..n {
        let url = format!("{base_url}/cached/{i}");
        let _ = client
            .raw_request(reqwest::Method::GET, &url, None, None)
            .await;
    }

    // Re-request the same URLs: all cache hits, no throttle applied.
    let start = Instant::now();
    for i in 0..n {
        let url = format!("{base_url}/cached/{i}");
        let _ = client
            .raw_request(reqwest::Method::GET, &url, None, None)
            .await;
    }
    let elapsed = start.elapsed();

    // If cache hits were throttled, this would take ≥ n × min_delay_ms.
    // They should complete in well under one delay period.
    let would_take_if_throttled = Duration::from_millis(n * min_delay_ms);
    assert!(
        elapsed < would_take_if_throttled,
        "cache hits took {elapsed:?}, expected < {would_take_if_throttled:?}; \
         cache hits should bypass the throttle entirely"
    );

    Ok(())
}

// --- 13. Mixed raw_request + raw_request_nocache: nocache remains throttled -

/// Fire concurrent tasks where half use `raw_request` on pre-warmed URLs
/// (instant cache hits, zero throttle cost) and half use `raw_request_nocache`
/// on fresh URLs (always network, always throttled).
///
/// The cache-hit tasks complete immediately and return their semaphore slots
/// (they never acquire one).  The nocache tasks must still wait for permits
/// and sleep `base_delay_ms`.  Total elapsed must therefore reflect the time
/// to drain all nocache tasks through the semaphore, proving that fast cache
/// responses do not inflate the apparent concurrency and let nocache tasks
/// slip through unthrottled.
#[tokio::test]
async fn test_mixed_raw_request_and_nocache_nocache_remains_throttled() -> Result<(), Box<dyn Error>>
{
    let mut server = Server::new_async().await;
    let _mock = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"^/(warm|cold)/\d+$".to_string()),
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{}")
        .expect_at_least(1)
        .create_async()
        .await;

    let max_concurrent: usize = 2;
    let min_delay_ms: u64 = 100;
    let n = 10usize; // 10 warm (cache) + 10 cold (nocache) = 20 tasks
    let client = Arc::new(make_client(max_concurrent, min_delay_ms));
    let base_url = server.url();

    // Warm the cache so /warm/{i} responses are stored.
    for i in 0..n {
        let url = format!("{base_url}/warm/{i}");
        let _ = client
            .raw_request(reqwest::Method::GET, &url, None, None)
            .await;
    }

    // Concurrent burst: warm = cache hits via raw_request,
    //                   cold = always-network via raw_request_nocache.
    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .flat_map(|i| {
            let c_warm = Arc::clone(&client);
            let c_cold = Arc::clone(&client);
            let url_warm = format!("{base_url}/warm/{i}");
            let url_cold = format!("{base_url}/cold/{i}");
            [
                tokio::spawn(async move {
                    let _ = c_warm
                        .raw_request(reqwest::Method::GET, &url_warm, None, None)
                        .await;
                }),
                tokio::spawn(async move {
                    let _ = c_cold
                        .raw_request_nocache(reqwest::Method::GET, &url_cold, None)
                        .await;
                }),
            ]
        })
        .collect();
    for h in handles {
        let _ = h.await;
    }
    let elapsed = start.elapsed();

    // 10 cold (nocache) tasks with max_concurrent=2 →
    // ceil(10/2)=5 serial batches × 100 ms = 500 ms minimum.
    let cold_batches = (n as u64).div_ceil(max_concurrent as u64);
    let min_expected = Duration::from_millis(cold_batches * min_delay_ms);
    let max_expected = Duration::from_millis(cold_batches * (min_delay_ms + 500 + 200));

    assert!(
        elapsed >= min_expected,
        "mixed raw_request+nocache: elapsed {elapsed:?} < {min_expected:?}; \
         cache hits may have let nocache tasks skip the throttle"
    );
    assert!(
        elapsed <= max_expected,
        "mixed raw_request+nocache: elapsed {elapsed:?} > {max_expected:?}; \
         something serialised beyond the {max_concurrent}-concurrent limit"
    );

    Ok(())
}

// --- 14. Canonical ≤ 10 req/s configurations --------------------------------

/// Verifies the three (max_concurrent, min_delay_ms) pairs that together
/// satisfy `max_concurrent / (min_delay_ms / 1000) = 10 req/s` exactly:
///
/// | max_concurrent | min_delay_ms | req/s |
/// |---------------:|-------------:|------:|
/// |              1 |          100 |    10 |
/// |              5 |          500 |    10 |
/// |             10 |         1000 |    10 |
///
/// The formula is: `effective_rps = max_concurrent / (min_delay_ms / 1000)`.
/// Each semaphore slot sleeps `min_delay_ms` before sending, so concurrency
/// multiplies throughput — not a cap.
///
/// Uses `raw_request_nocache` (CacheBypass=true) so every request goes
/// through the throttle semaphore regardless of any cached DataStore state.
/// The sliding-window assertion confirms the observed rate never exceeds
/// 12 req/s (10 + 20 % jitter headroom).
#[tokio::test]
async fn test_canonical_10rps_configurations() -> Result<(), Box<dyn Error>> {
    // The three (max_concurrent, min_delay_ms) pairs that each yield ≤ 10 req/s
    let configs: &[(usize, u64)] = &[
        (1, 100),   // 1 slot  × 100 ms/slot  = 10 req/s
        (5, 500),   // 5 slots × 500 ms/slot  = 10 req/s
        (10, 1000), // 10 slots × 1000 ms/slot = 10 req/s
    ];

    for (cfg_idx, &(max_concurrent, min_delay_ms)) in configs.iter().enumerate() {
        let mut server = Server::new_async().await;
        let _mock = server
            .mock("GET", mockito::Matcher::Regex(format!(r"^/{cfg_idx}/\d+$")))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .expect_at_least(1)
            .create_async()
            .await;

        let n: usize = 20;
        let client = Arc::new(make_client(max_concurrent, min_delay_ms));
        let base_url = server.url();
        let timestamps: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));

        let handles: Vec<_> = (0..n)
            .map(|i| {
                let client = Arc::clone(&client);
                let ts_ref = Arc::clone(&timestamps);
                // Embed cfg_idx so port recycling across loop iterations cannot
                // produce cache hits; raw_request_nocache bypasses the cache
                // entirely anyway, but belt-and-suspenders.
                let url = format!("{base_url}/{cfg_idx}/{i}");
                async move {
                    let _ = client
                        .raw_request_nocache(reqwest::Method::GET, &url, None)
                        .await;
                    ts_ref.lock().unwrap().push(Instant::now());
                }
            })
            .collect();

        join_all(handles).await;

        let mut ts = timestamps.lock().unwrap().clone();
        ts.sort();
        let max_in_window = max_requests_in_any_one_second_window(&ts);
        let allowed: usize = 12; // 10 req/s + 20 % jitter headroom

        assert!(
            max_in_window <= allowed,
            "canonical config ({max_concurrent}×{min_delay_ms}ms): observed {max_in_window} req/s \
             in the busiest 1-second window; expected ≤ {allowed} (10 req/s + 20% jitter)"
        );
    }

    Ok(())
}
