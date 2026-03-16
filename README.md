> **Experimental research project.** Not affiliated with or endorsed by the SEC. Not investment advice. Use at your own risk.

[deepwiki-page]: https://deepwiki.com/jzombie/rust-sec-fetcher
[deepwiki-badge]: https://deepwiki.com/badge.svg

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

Defaults are conservative (2 req/s, 1 concurrent request) to stay well within the SEC's 10 req/s limit. Override with `min_delay_ms`, `max_concurrent`, and `max_retries` in the config file.

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Free for personal and academic research. **Commercial use, and use as AI training data, are prohibited** without a separate agreement — contact info@zenosmosis.com.
