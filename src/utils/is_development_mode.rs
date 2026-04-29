pub fn is_development_mode() -> bool {
    std::env::var("CARGO_PROFILE")
        .map(|profile| profile == "dev")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_development_mode_default_false() {
        // In test context, CARGO_PROFILE is not set to "dev"
        assert!(!is_development_mode());
    }

    #[test]
    fn test_is_development_mode_with_dev_profile() {
        unsafe { std::env::set_var("CARGO_PROFILE", "dev") };
        assert!(is_development_mode());
        unsafe { std::env::remove_var("CARGO_PROFILE") };
    }

    #[test]
    fn test_is_development_mode_with_release_profile() {
        unsafe { std::env::set_var("CARGO_PROFILE", "release") };
        assert!(!is_development_mode());
        unsafe { std::env::remove_var("CARGO_PROFILE") };
    }
}
