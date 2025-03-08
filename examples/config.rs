use sec_fetcher::config::ConfigManager;

fn main() {
    let suggested_system_path = ConfigManager::get_suggested_system_path();
    println!("Suggested system path: {:?}", suggested_system_path);

    let config_path = ConfigManager::get_config_path();
    println!("Config path: {:?}", config_path);

    let config_manager = ConfigManager::load().unwrap();
    let cache_mode = config_manager.get_cache_mode();

    println!("Cache mode: {:?}", cache_mode);
}
