use sec_fetcher::accessor::get_us_gaap_human_readable_mapping;

pub fn main() {
    println!(
        "{:?}",
        get_us_gaap_human_readable_mapping("LiabilitiesAndPartnersCapital")
    );
}
