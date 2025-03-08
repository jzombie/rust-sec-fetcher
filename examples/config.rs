use sec_fetcher::config::{ConfigManager, CredentialManager};

fn main() {
    let suggested_system_path = ConfigManager::get_suggested_system_path();
    println!("Suggested system path: {:?}", suggested_system_path);

    let config_path = ConfigManager::get_config_path();
    println!("Config path: {:?}", config_path);

    // let config_manager = ConfigManager::load(Some(&credential_manager)).map_err(|err| eprint!("{:?}\n", err)).unwrap();
    let config_manager = ConfigManager::load().unwrap();

    let config = config_manager.get_config();
    print!("{}\n", config.pretty_print());

    let cache_mode = config.get_http_cache_mode();
    let cache_dir = config.get_http_cache_dir();

    println!("Cache mode: {:?}", cache_mode);
    println!("Cache dir: {:?}", cache_dir);
}
