TODO: Keep notice of 10-Q/A and 10-K/A forms (these are amended)

## SIC

- SIC code list: https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list
- SIC code lookup: https://data.sec.gov/submissions/CIK##########.json

Note: Instead of manually maintaining a SIC list, autogenerate it when doing CIK lookups.

```
via: https://data.sec.gov/submissions/CIK##########.json
pub sic: Option<u64>,                // i.e. 3571
pub sic_description: Option<String>, // i.e. "Electronic Computers"
```

```
"sic": "XXXX"
```

## Sector Mappings

The **11-sector classification** often used in financial markets and economic analysis maps SIC codes into the following broad categories:

The **SIC codes less than 1000** primarily cover **agriculture, forestry, fishing, and related services**. These industries typically fall under the **Materials** or **Consumer Staples** sectors, depending on their classification.

### **Mapping SIC Codes < 1000 to Sectors**
| **SIC Code Range** | **Industry**                        | **Mapped Sector**    |
|--------------------|-------------------------------------|----------------------|
| **0100-0199**      | Agricultural Production – Crops     | **Consumer Staples** |
| **0200-0299**      | Agricultural Production – Livestock | **Consumer Staples** |
| **0700-0799**      | Agricultural Services               | **Industrials**      |
| **0800-0899**      | Forestry                            | **Materials**        |
| **0900-0999**      | Fishing, Hunting, and Trapping      | **Consumer Staples** |

### **Sector Justification**
- **Consumer Staples**: Covers food-related agriculture (e.g., crops, livestock, and fisheries).
- **Materials**: Covers raw material-related industries such as forestry and logging.
- **Industrials**: Covers agricultural support services like farm machinery and commercial agricultural operations.

Would you like a **complete SIC-to-sector mapping function** for automatic classification?

### **11 Common Sectors with SIC Mapping**
| **Sector**                     | **SIC Code Range**                                                                                |
|--------------------------------|---------------------------------------------------------------------------------------------------|
| **Energy**                     | 1000-1499, 2900-2999, 4900-4999                                                                   |
| **Materials**                  | 2800-2899, 3200-3299, 3300-3399                                                                   |
| **Industrials**                | 1500-1799, 3400-3499, 3500-3599, 3700-3799                                                        |
| **Consumer Discretionary**     | 2300-2399, 2500-2599, 2700-2799, 3100-3199, 3900-3999, 5000-5099, 5600-5699, 5700-5799, 5900-5999 |
| **Consumer Staples**           | 2000-2099, 2100-2199, 5400-5499, 5500-5599                                                        |
| **Health Care**                | 8000-8099                                                                                         |
| **Financials**                 | 6000-6799                                                                                         |
| **Information Technology**     | 3570-3579, 3600-3699, 7370-7379                                                                   |
| **Telecommunication Services** | 4800-4899                                                                                         |
| **Utilities**                  | 4900-4999                                                                                         |
| **Real Estate**                | 6500-6799                                                                                         |

### **Notes on the Mapping**

- **Energy** covers oil, gas, and utilities.
- **Materials** includes chemicals, metals, and mining.
- **Industrials** spans machinery, construction, and manufacturing.
- **Consumer Discretionary** includes luxury goods, retail, and entertainment.
- **Consumer Staples** consists of essential goods like food and beverages.
- **Health Care** includes hospitals, pharmaceuticals, and medical equipment.
- **Financials** covers banks, insurance, and investment services.
- **Information Technology** includes computing, semiconductors, and software.
- **Telecommunications** consists of phone, internet, and broadcasting.
- **Utilities** includes power, gas, and water providers.
- **Real Estate** covers property management, REITs, and land leasing.



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

Filings -> recent -> primary document -> NPORT-P -> accession number

```
https://www.sec.gov/Archives/edgar/data/884394/000175272425043826/primary_doc.xml
```

### Parsing ETFs from Funds

- Funds will commonly have `ETF` in their name to designate they are indeed an ETF.
- Title may say "EXCHANGE-TRADED FUND"

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

### N-PORT Field Explanations:

| **Field**                         | **Meaning**                                                                                                                                                                         | **Possible Values**                                                                                           |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `<name>`                          | The name of the **issuing entity** (company, fund, etc.).                                                                                                                           | A string, typically the legal name of the company (e.g., `"Everest Group Ltd."`).                             |
| `<lei>`                           | **Legal Entity Identifier (LEI)** – a unique global identifier for legal entities engaged in financial transactions.                                                                | A 20-character alphanumeric string assigned to the company (e.g., `"549300N24XF2VV0B3570"`).                  |
| `<title>`                         | The security's title, usually the same as `<name>`.                                                                                                                                 | A string, often the same as `<name>`.                                                                         |
| `<cusip>`                         | **CUSIP (Committee on Uniform Securities Identification Procedures)** – a 9-character alphanumeric identifier for securities traded in the U.S.                                     | A valid CUSIP or `"N/A"` if unavailable (e.g., `"N/A"`).                                                      |
| `<identifiers>`                   | A container for security identifiers such as **ISIN (International Securities Identification Number)**.                                                                             | Contains one or more identifiers (e.g., `<isin value="BMG3223R1088"/>`).                                      |
| `<isin>` (Inside `<identifiers>`) | **ISIN (International Securities Identification Number)** – a unique identifier for a security, consisting of a 2-letter country code, a 9-character identifier, and a check digit. | A 12-character alphanumeric code (e.g., `"BMG3223R1088"`).                                                    |
| `<balance>`                       | The **quantity of the security held**.                                                                                                                                              | A decimal number representing the number of shares or units (e.g., `"193.00000000"`).                         |
| `<units>`                         | The **unit type** of the holding.                                                                                                                                                   | `"NS"` (likely **Number of Shares**), or other unit types like `"Bonds"`, `"Contracts"`, etc.                 |
| `<curCd>`                         | The **currency code** of the security's valuation.                                                                                                                                  | A 3-letter ISO 4217 currency code (e.g., `"USD"`).                                                            |
| `<valUSD>`                        | The **market value** of the holding in **USD**.                                                                                                                                     | A decimal number representing the total value in USD (e.g., `"68632.73000000"`).                              |
| `<pctVal>`                        | The **percentage of total portfolio value** that this security represents.                                                                                                          | A decimal number representing the percentage (e.g., `"0.039377694831"` for ~0.0394%).                         |
| `<payoffProfile>`                 | The **investment position type**.                                                                                                                                                   | `"Long"` (holding the asset expecting appreciation) or `"Short"` (betting on depreciation).                   |
| `<assetCat>`                      | The **asset class category** of the security.                                                                                                                                       | `"EC"` (Equity), `"FI"` (Fixed Income), `"RE"` (Real Estate), `"HY"` (High Yield), `"CM"` (Commodities), etc. |
| `<issuerCat>`                     | The **issuer classification**.                                                                                                                                                      | `"CORP"` (Corporate), `"GOV"` (Government), `"MUNI"` (Municipal), etc.                                        |
| `<invCountry>`                    | The **country of investment** or incorporation of the issuer.                                                                                                                       | A 2-letter country code (ISO 3166-1 alpha-2, e.g., `"BM"` for Bermuda).                                       |
| `<isRestrictedSec>`               | Indicates if the security is **restricted** (not freely tradable).                                                                                                                  | `"Y"` (Yes, restricted) or `"N"` (No, freely tradable).                                                       |
| `<fairValLevel>`                  | **Fair Value Level** per **FASB ASC 820 (GAAP's Fair Value Hierarchy)**.                                                                                                            | `1` (Market Price), `2` (Observable Inputs), `3` (Unobservable Inputs).                                       |
| `<securityLending>`               | A container for security lending-related details.                                                                                                                                   | Contains boolean fields for lending activity.                                                                 |
| `<isCashCollateral>`              | Indicates if **cash collateral** is involved in a securities lending agreement.                                                                                                     | `"Y"` (Yes) or `"N"` (No).                                                                                    |
| `<isNonCashCollateral>`           | Indicates if **non-cash collateral** is involved (e.g., securities, bonds).                                                                                                         | `"Y"` (Yes) or `"N"` (No).                                                                                    |
| `<isLoanByFund>`                  | Indicates if the fund itself has **loaned out the security**.                                                                                                                       | `"Y"` (Yes, loaned) or `"N"` (No).                                                                            |


## TODO: Company information via latest 10-K

submissions CIK -> Latest 10-K -> primary document

Since primary documents for 10-Ks are .htm files, you need to:

- Retrieve the filing URL.
- Download the HTML document of the primary document.
- Summarize the business section.

## US-GAAP Mappings

- https://www.sec.gov/files/edgar/filer-information/specifications/xbrl-guide-2024-07-08.pdf
- http://xbrl.squarespace.com/understanding-sec-xbrl-financi/

### Human-Readable Values and Impute

- http://www.xbrlsite.com/2014/Reference/Mapping.pdf
- http://www.xbrlsite.com/2014/Reference/ImputeRules.pdf

### Third-Party Assessments / Formulas

- https://documentation.alphavantage.co/FundamentalDataDocs/gaap_documentation.html
- https://polygon.io/knowledge-base/article/what-fields-can-i-expect-from-polygons-financials-api
- https://site.financialmodelingprep.com/developer/docs/formula


## Other Third-Party

### Prices

- https://docs.alpaca.markets/docs/about-market-data-api

### Logos

- https://www.logo.dev/pricing

### Related

- https://sec-edgar-api.readthedocs.io/en/latest/
- https://github.com/janlukasschroeder/sec-api-python

-----

This project is not affiliated with or endorsed by the U.S. Securities and Exchange Commission.
