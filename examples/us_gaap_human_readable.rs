use sec_fetcher::transformers::distill_us_gaap_fundamental_concepts;

pub fn main() {
    println!(
        "{:?}",
        distill_us_gaap_fundamental_concepts("LiabilitiesAndPartnersCapital")
    );
}
