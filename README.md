**This is an experimental research project. Documentation is currently not provided.**

[deepwiki-page]: https://deepwiki.com/jzombie/rust-sec-fetcher
[deepwiki-badge]: https://deepwiki.com/badge.svg

## ⚠️ Disclaimers

**Not Affiliated with the SEC:** This project is not affiliated with, endorsed by, or associated with the U.S. Securities and Exchange Commission (SEC) in any way.

**Not Investment Advice:** This software is an experimental research tool for data retrieval only. Nothing produced by this tool should be construed as investment, financial, legal, or tax advice. Stock decisions and financial analysis should be grounded in your own independent research and professional consultation.

**"As-Is" Data Accuracy:** This tool parses data from external sources (SEC EDGAR). I make no guarantees regarding the accuracy, completeness, timeliness, or reliability of the data retrieved.

- Parsing logic may fail due to changes in SEC filing formats (XBRL/HTML).

- Data may be delayed or omitted due to network or upstream server issues.

**Limitation of Liability:** By using this software, you acknowledge that you do so at your own risk. In no event shall the author be held liable for any financial losses, trading errors, or damages resulting from the use or inability to use this tool, even if advised of the possibility of such damages.

## Configuration

Configuration is loaded from `sec_fetcher_config.toml` in the working directory (or the platform config directory). Copy `sec_fetcher_config.toml.example` as a starting point.

### Required: `email`

The SEC requires every automated EDGAR request to carry a `User-Agent` header containing a contact email address, so that the SEC can reach out if a client causes problems. See [Accessing EDGAR Data](https://www.sec.gov/os/accessing-edgar-data).

Without a valid email address this library will refuse to make any network requests.

> **Privacy note:** Your email address is included only in the `User-Agent` header sent to SEC EDGAR servers — it is not transmitted to any other party. You are responsible for selecting the most secure method of supplying this value that is appropriate for your own use case (e.g. environment variable, config file with restricted permissions, etc.).

The email is resolved in this order (highest precedence first):

| #   | Source               | Example                                                  |
| --- | -------------------- | -------------------------------------------------------- |
| 1   | Config file          | `email = "you@example.com"` in `sec_fetcher_config.toml` |
| 2   | Environment variable | `SEC_FETCHER_EMAIL=you@example.com`                      |
| 3   | Interactive prompt   | shown automatically when `stdin`/`stdout` are a terminal |

If none of these is available the program exits with an error.

### Rate limiting

The SEC specifies a maximum of [10 requests/second](https://www.sec.gov/os/accessing-edgar-data). The default configuration is intentionally conservative at **2 requests/second** (500 ms minimum delay, 1 concurrent request). Override in the config file:

```toml
max_concurrent = 1   # concurrent in-flight requests
min_delay_ms   = 500 # ms gap between requests
max_retries    = 3
```

## Licensing

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

### For Personal & Research Use

You are free to use, modify, and distribute this software for personal or academic research projects, provided that:

- The original license and copyright notices remain intact.

- Any modified distributions point back to the original creator and this license.

### For Professional & AI Usage

**Commercial and professional use is strictly prohibited under the default license.** This includes, but is not limited to:

- Use within a corporate or for-profit environment.

- Use as training data, input, or fine-tuning material for AI/Machine Learning models.

### Integration into paid products or services

**Custom Licensing**: Professional usage and AI-related rights are considered on a case-by-case basis and may require a separate commercial agreement or payment. To request a commercial waiver, please contact info@zenosmosis.com.

### Compliance Note

Users of this tool are responsible for adhering to the [SEC.gov External Data Access Policy](https://www.sec.gov/os/accessing-edgar-data). This tool includes default rate-limiting to help stay within the 10 requests/second limit, but use in a commercial high-frequency environment requires a separate agreement to ensure technical and legal compliance.
