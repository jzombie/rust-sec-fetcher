use sec_fetcher::config::ConfigManager;
use std::path::PathBuf;

fn main() {
    // let suggested_system_path = ConfigManager::get_suggested_system_path();
    // println!("Suggested system path: {:?}", suggested_system_path);

    // let config_path = ConfigManager::get_config_path();
    // println!("Config path: {:?}", config_path);

    // let config_manager = ConfigManager::from_config(Some(PathBuf::from("invalid"))).unwrap();

    let config_manager = ConfigManager::load().unwrap();

    let config = config_manager.get_config();
    print!("{}\n", config.pretty_print());
}
