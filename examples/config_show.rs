use clap::Parser;
use sec_fetcher::config::ConfigManager;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Display the active sec-fetcher configuration")]
struct Args {
    /// Path to a specific config file (overrides the default discovery order)
    #[arg(long)]
    config_path: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    let config_manager = if let Some(path) = args.config_path {
        ConfigManager::from_config(Some(path)).unwrap()
    } else {
        ConfigManager::load().unwrap()
    };

    let config = config_manager.get_config();
    println!("{}", config.pretty_print());
}

