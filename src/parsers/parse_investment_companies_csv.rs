use csv::ReaderBuilder;
use std::io::Cursor;
use crate::models::InvestmentCompany;
use std::error::Error;
use bytes::Bytes;

pub fn parse_investment_companies_csv(byte_array: Bytes) -> Result<Vec<InvestmentCompany>, Box<dyn Error>> {
    let cursor = Cursor::new(&byte_array);
    let mut reader = ReaderBuilder::new().from_reader(cursor);

    let mut records = Vec::new();

    for result in reader.deserialize() {
        let record: InvestmentCompany = result?;
        records.push(record);
    }

    Ok(records)
}
