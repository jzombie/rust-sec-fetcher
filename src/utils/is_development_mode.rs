pub fn is_development_mode() -> bool {
    std::env::var("CARGO_PROFILE")
        .map(|profile| profile == "dev")
        .unwrap_or(false)
}
