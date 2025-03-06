TODO: Keep notice of 10-Q/A and 10-K/A forms (these are amended)

- SIC code list: https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list
- SIC code lookup: https://data.sec.gov/submissions/CIK##########.json
```
"sic": "XXXX"
```

## Constructing ETF Constituent Lookups

You can access **Form N-PORT filings** through the SEC's structured data endpoints programmatically via their public JSON API.

### How to access Form N-PORT via SEC API:

The SEC API endpoint for filings is structured as follows:

```http
https://data.sec.gov/submissions/CIK##########.json
```

Replace `##########` with the ETF's **CIK** number (10 digits, padded with zeros).

---

### Example API call (SPY - SPDR S&P 500 ETF):

- **SPY ETF CIK**: `0000884394`

**API URL**:
```http
https://data.sec.gov/submissions/CIK0000884394.json
```

From this JSON:

- Check the `filings -> recent` object.
- Filter `form` fields for `"NPORT-P"` to locate recent Form N-PORT filings.
- Use the provided accession numbers to construct direct URLs to retrieve filings.

### Retrieving actual N-PORT filing documents:

With the accession number from the JSON (`"accessionNumber": "0001752724-24-055725"`), construct URLs like this:

```http
https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION-NUMBER}/
```

For example, SPY:

```
https://www.sec.gov/Archives/edgar/data/0000884394/000175272424055725/
```

Within this directory, you'll find XML or HTML files containing ETF holdings.

TODO: See if there is always a primary doc like this:

```
https://www.sec.gov/Archives/edgar/data/884394/000175272425043826/primary_doc.xml
```


---

### Important Notes:

- The SEC expects API users to include a `User-Agent` header with your contact information (e.g., email):
  ```http
  User-Agent: MyAppName/1.0 (myemail@example.com)
  ```

- SEC data is **rate-limited** (typically <10 requests/sec).

---

### Investment Company Series and Class Information

```
https://www.sec.gov/about/opendatasetsshtmlinvestment_company
```

CSV download URLs like:

```
https://www.sec.gov/files/investment/data/other/investment-company-series-and-class-information/investment-company-series-class-2024.csv
```

The SEC’s Investment Company Series and Class dataset doesn't explicitly label ETFs directly. However, ETFs can generally be identified based on common characteristics in the dataset:

Common ways to determine ETFs:
Class Name or Series Name
Look for the terms:

"ETF"
"Exchange-Traded Fund"
"Trust"
"Shares"
ETFs typically contain these keywords explicitly in their Series Name or Class Name.

Entity Name
Common ETF issuers have recognizable names:

"iShares"
"SPDR"
"Vanguard"
"Invesco"
"State Street"
The Entity Name column will reflect these ETF providers.

## Logo API

- https://www.logo.dev/pricing
