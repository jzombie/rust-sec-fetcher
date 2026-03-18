# sec-fetcher

[![made-with-rust][rust-logo]][rust-src-page] [![crates.io][crates-badge]][crates-page] [![PolyForm NC 1.0.0 licensed][polyform-license-badge]][polyform-license-page] [![DeepWiki][deepwiki-badge]][deepwiki-page] [![Coverage][coveralls-badge]][coveralls-page]

`sec-fetcher` is a Rust library for programmatic access to SEC EDGAR: resolve tickers to CIKs, fetch company filings (10-K, 10-Q, 8-K, and more), render filing documents as clean text or Markdown, track fund holdings across N-PORT and 13F filings, monitor IPO registrations, and process bulk US GAAP XBRL datasets — all with configurable rate limiting designed to stay within SEC usage guidelines.
>
> **Experimental research project.** Not affiliated with or endorsed by the U.S. Securities and Exchange Commission. Not investment advice. Use at your own risk.

Before running examples, complete [setup and configuration](#configuration).

## Examples

All examples require a [configured email address](#email-required). Set
`SEC_FETCHER_EMAIL=your@email.com` in your environment to get started.

### Client setup

Every program that uses this library starts by loading the configuration and
constructing a rate-limited HTTP client:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::SecClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;
    // pass &client to any fetch_* function below
    Ok(())
}
```

> `cargo run --example config_show` prints the active configuration so you
> can confirm your email and rate-limit settings before running anything else.

---

### Ticker & company lookup

Resolve a ticker to a CIK and fetch the full SEC company profile:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_company_profile, SecClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let ticker = TickerSymbol::new("AAPL");
    let cik = fetch_cik_by_ticker_symbol(&client, &ticker).await?;
    let profile = fetch_company_profile(&client, cik).await?;

    println!("{} (CIK {})", profile.name, profile.cik);
    println!("Sector:    {:?}", profile.sector());
    println!("SIC:       {:?}", profile.sic);
    println!("Exchanges: {}", profile.exchanges.join(", "));
    Ok(())
}
```

Fuzzy-match a company name against the full SEC operating-company list:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::Ticker;
use sec_fetcher::network::{fetch_company_tickers, SecClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let all = fetch_company_tickers(&client, false).await?;
    let matches = Ticker::get_by_fuzzy_matched_name(&all, "Eli Lilly", None);
    println!("{:?}", matches);
    Ok(())
}
```

> Runnable examples: `cargo run --example ticker_list`,
> `cargo run --example ticker_show -- AAPL`,
> `cargo run --example company_show -- AAPL MSFT NVDA`,
> `cargo run --example company_search -- "Eli Lilly"`,
> `cargo run --example fuzzy_match_company -- "johnson johnson"`,
> `cargo run --example cik_show -- SPY`

---

### Filings

Fetch and render the latest 10-K — body and all substantive exhibits:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{fetch_cik_by_ticker_symbol, fetch_filings, SecClient};
use sec_fetcher::ops::render_filing;
use sec_fetcher::views::MarkdownView;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("AAPL")).await?;
    let filings = fetch_filings(&client, cik, "10-K").await?;

    // render_filing(client, filing, render_body, render_exhibits, view)
    let rendered = render_filing(&client, &filings[0], true, true, &MarkdownView).await?;

    if let Some(body) = rendered.body {
        println!("{}", body);
    }
    for exhibit in &rendered.exhibits {
        println!("--- {} ---\n{}", exhibit.document_type, exhibit.content);
    }
    Ok(())
}
```

List all 8-K filings for a ticker and their primary document URLs:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{fetch_8k_filings, fetch_cik_by_ticker_symbol, SecClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("LLY")).await?;
    let filings = fetch_8k_filings(&client, cik).await?;

    for f in &filings {
        println!(
            "{:?}  [{}]  {}",
            f.filing_date,
            f.items.join(", "),
            f.as_primary_document_url()
        );
    }
    Ok(())
}
```

Render any EDGAR document URL directly (HTML → clean text):

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_and_render, SecClient};
use sec_fetcher::views::EmbeddingTextView;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let url = "https://www.sec.gov/Archives/edgar/data/320193/\
               000032019325000008/aapl-20241228.htm";
    let text = fetch_and_render(&client, url, &EmbeddingTextView).await?;
    println!("{}", text);
    Ok(())
}
```

> Runnable examples: `cargo run --example filing_show -- AAPL --form 10-K`,
> `cargo run --example filing_show -- LLY --form 10-Q --view markdown`,
> `cargo run --example 8k_list -- LLY`,
> `cargo run --example 8k_exhibits_as_markdown -- AAPL --view markdown`,
> `cargo run --example press_release_show -- MSFT --earnings-only`,
> `cargo run --example filing_render -- <URL> --view markdown`

---

### EDGAR feed & index

Delta-poll the live EDGAR Atom feed — newest filings first. Persist
`delta.high_water` and pass it as `since` on the next call to receive only new
filings with no gaps and no duplicates:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_edgar_feeds_since, SecClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    // One page = 40 entries. Pass `since` on subsequent calls for delta polling.
    let delta = fetch_edgar_feeds_since(&client, &["8-K"], None, 1).await?;
    for entry in &delta.entries {
        println!(
            "{:?}  {}  {}",
            entry.filing_date, entry.form_type, entry.company_name
        );
    }
    // delta.high_water is the timestamp to pass as `since` next time.
    Ok(())
}
```

Query EDGAR's historical quarterly full-index — every filing since Q4 1993:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::network::{fetch_edgar_master_index, SecClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let entries = fetch_edgar_master_index(&client, 2024, 4).await?;
    let ten_ks: Vec<_> = entries.iter().filter(|e| e.form_type == "10-K").collect();
    println!("{} annual reports filed in Q4 2024", ten_ks.len());
    Ok(())
}
```

> Runnable examples: `cargo run --example edgar_feed_poll -- --filter "8-K" --pages 5`,
> `cargo run --example edgar_feed_poll -- --filter "8-K" --since "2026-03-13T17:30:01-04:00"`,
> `cargo run --example edgar_index_browse -- --form 10-K --year 2024 --quarter 4`

---

### Holdings & funds

Fetch the latest N-PORT portfolio for an ETF:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::{CikSubmission, TickerSymbol};
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_cik_submissions, fetch_nport, SecClient,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("SPY")).await?;
    let submissions = fetch_cik_submissions(&client, cik).await?;
    let latest = CikSubmission::by_form(&submissions, "NPORT-P")
        .into_iter()
        .next()
        .ok_or("no NPORT-P filings found")?;

    let holdings = fetch_nport(&client, latest).await?;
    for h in holdings.iter().take(10) {
        println!("{:9}  {:<40}  {:.4}%", h.cusip, h.name, h.pct_val);
    }
    Ok(())
}
```

Diff two consecutive N-PORT filings to see what a fund bought and sold:

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::models::TickerSymbol;
use sec_fetcher::network::{
    fetch_cik_by_ticker_symbol, fetch_nport, fetch_nport_filings, SecClient,
};
use sec_fetcher::ops::{diff_holdings, positions_from_nport};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let cik = fetch_cik_by_ticker_symbol(&client, &TickerSymbol::new("QQQ")).await?;
    // Filings are returned newest-first.
    let filings = fetch_nport_filings(&client, cik).await?;

    let newer = positions_from_nport(&fetch_nport(&client, &filings[0]).await?);
    let older = positions_from_nport(&fetch_nport(&client, &filings[1]).await?);
    let diff = diff_holdings(&older, &newer);

    println!("{} positions added", diff.added.len());
    println!("{} positions removed", diff.removed.len());
    println!("{} weight changes ≥ threshold", diff.changed.len());
    Ok(())
}
```

> Runnable examples: `cargo run --example holdings_show -- SPY`,
> `cargo run --example holdings_show -- BRK-A` (13F),
> `cargo run --example holdings_show -- AAPL` (Form 4 insider transactions),
> `cargo run --example nport_render -- QQQ`,
> `cargo run --example fund_series_list`

---

### IPO tracking

Scan the EDGAR feed for new IPO registration filings (S-1, S-1/A, F-1, F-1/A):

```rust,no_run
use sec_fetcher::config::ConfigManager;
use sec_fetcher::enums::FormType;
use sec_fetcher::network::SecClient;
use sec_fetcher::ops::get_ipo_feed_entries;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigManager::load()?;
    let client = SecClient::from_config_manager(&config)?;

    let form_types: Vec<FormType> = FormType::IPO_REGISTRATION_FORM_TYPES.to_vec();
    // Scan 5 pages × 40 entries = up to 200 recent feed entries.
    let (entries, _high_water) = get_ipo_feed_entries(&client, &form_types, None, 5).await?;

    for e in &entries {
        println!(
            "{:?}  {}  {}  (CIK {:?})",
            e.filing_date,
            e.form_type,
            e.company_name,
            e.cik.as_ref().map(|c| c.value)
        );
    }
    Ok(())
}
```

> Runnable examples: `cargo run --example ipo_list -- --pages 10`,
> `cargo run --example ipo_show -- --ticker RDDT --part summary`,
> `cargo run --example ipo_show -- --cik 1713445 --index -1 --part body`

---

### US GAAP data (local bulk files)

The `pull-us-gaap-bulk` binary downloads the SEC's bulk XBRL dataset and writes
one CSV per ticker to `data/`. Once downloaded, the GAAP examples work entirely
offline:

```sh
# Search every CSV for a given XBRL concept:
cargo run --example us_gaap_search -- Assets
cargo run --example us_gaap_search -- NetIncomeLoss --max-values 5

# Ranked frequency table of XBRL concept names across the full dataset:
cargo run --example us_gaap_column_stats
```

## Feature flags

| Feature   | Default  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `keyring` | disabled | Enables OS keychain integration so the SEC contact email can be stored and retrieved securely across sessions (macOS Keychain, Windows Credential Manager, Linux Secret Service via D-Bus). When this feature is active and no email is found in the config file or environment, the library will prompt for one interactively and remember it. Without this feature the email must always be supplied via the config file or `SEC_FETCHER_EMAIL` environment variable. |

### Linux note

The `keyring` feature depends on D-Bus and requires `libdbus-1-dev` and `pkg-config` to be installed at build time:

```sh
# Ubuntu / Debian
sudo apt install libdbus-1-dev pkg-config

# Fedora / RHEL
sudo dnf install dbus-devel pkgconf-pkg-config
```

Enable the feature with:

```sh
cargo build --features keyring
cargo test --features keyring
```

No extra system packages are required on macOS or Windows.

## Configuration

No config file is required to get started. The simplest way to try it out is to just run the binary — if you are in a terminal and no email is configured, the program will ask for it at startup. For repeated use you can set the `SEC_FETCHER_EMAIL` environment variable, or copy `sec_fetcher_config.toml.example` to `sec_fetcher_config.toml` to keep a persistent config file.

### Email (required)

The SEC mandates a contact address in every automated request's `User-Agent` header ([policy](https://www.sec.gov/os/accessing-edgar-data)). The email is resolved in this order:

1. **Config file** — `email = "your.name@example.com"` in `sec_fetcher_config.toml`
2. **Environment variable** — `SEC_FETCHER_EMAIL=your@example.com cargo run`
3. **Startup prompt** — if neither of the above is set and you are running in a terminal, the program will ask you to type your email before it does anything else

### App name and version override (optional)

The first two segments of the `User-Agent` string sent to the SEC are the app name and version (`AppName/Version (+email)`). Both default to sec-fetcher's own values at runtime. Override either one in any of these ways:

|         | Programmatic string override                                                 | Config file             | Environment variable            |
| ------- | ----------------------------------------------------------------------------- | ----------------------- | ------------------------------- |
| Name    | `ConfigManager::from_config_with_app_identity(path, Some("my-app"), None)` | `app_name = "my-app"`   | `SEC_FETCHER_APP_NAME=my-app`   |
| Version | `ConfigManager::from_config_with_app_identity(path, None, Some("1.2.3"))`  | `app_version = "1.2.3"` | `SEC_FETCHER_APP_VERSION=1.2.3` |

Precedence for these two fields is:

1. **Programmatic string override** (from `from_config_with_app_identity`)
2. **Config file** (`app_name` / `app_version`)
3. **Environment variable** (`SEC_FETCHER_APP_NAME` / `SEC_FETCHER_APP_VERSION`)
4. **Built-in defaults** (`sec-fetcher` / crate version)

### Rate limiting

The SEC's public guidance limits automated requests to **10 req/s** ([policy](https://www.sec.gov/os/accessing-edgar-data)).

Throttle behaviour is controlled by two config values:

- `max_concurrent` — number of simultaneous in-flight requests (semaphore slots)
- `min_delay_ms` — minimum sleep each slot applies before sending

Effective throughput = `max_concurrent ÷ (min_delay_ms ÷ 1000)` req/s. Because each slot sleeps `min_delay_ms` before sending and holds its permit for the full round-trip, concurrency multiplies throughput — it is not a cap.

The three canonical configurations that each deliver **exactly 10 req/s**:

| `max_concurrent` | `min_delay_ms` | Effective req/s |
| ---------------: | -------------: | --------------: |
|                1 |            100 |              10 |
|                5 |            500 |              10 |
|               10 |           1000 |              10 |

The **default** (`max_concurrent = 1, min_delay_ms = 500`) delivers **2 req/s**, which is conservative and well under the SEC limit.

> **Note:** There is no built-in hard cap at 10 req/s. Setting `max_concurrent = 10, min_delay_ms = 50` would produce 200 req/s. The library does exactly what you configure — staying within the SEC limit is your responsibility.

Override the defaults with `min_delay_ms`, `max_concurrent`, and `max_retries` in the config file.

## License

Licensed under the [PolyForm Noncommercial License 1.0.0][polyform-license-page]. Free for personal and academic research. **Commercial use, and use as AI training data, are prohibited** without a separate agreement — contact info@zenosmosis.com.

[rust-src-page]: https://www.rust-lang.org/
[rust-logo]: https://img.shields.io/badge/Made%20with-Rust-black

[crates-page]: https://crates.io/crates/sec-fetcher
[crates-badge]: https://img.shields.io/crates/v/sec-fetcher.svg

[polyform-license-page]: https://polyformproject.org/licenses/noncommercial/1.0.0/
[polyform-license-badge]: https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue.svg

[coveralls-page]: https://coveralls.io/github/jzombie/rust-sec-fetcher?branch=main
[coveralls-badge]: https://img.shields.io/coveralls/github/jzombie/rust-sec-fetcher

[deepwiki-page]: https://deepwiki.com/jzombie/rust-sec-fetcher
[deepwiki-badge]: https://deepwiki.com/badge.svg
