> **Experimental research project.** Not affiliated with or endorsed by the SEC. Not investment advice. Use at your own risk.

[deepwiki-page]: https://deepwiki.com/jzombie/rust-sec-fetcher
[deepwiki-badge]: https://deepwiki.com/badge.svg

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

|         | Config file             | Environment variable            |
| ------- | ----------------------- | ------------------------------- |
| Name    | `app_name = "my-app"`   | `SEC_FETCHER_APP_NAME=my-app`   |
| Version | `app_version = "1.2.3"` | `SEC_FETCHER_APP_VERSION=1.2.3` |

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

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Free for personal and academic research. **Commercial use, and use as AI training data, are prohibited** without a separate agreement — contact info@zenosmosis.com.
