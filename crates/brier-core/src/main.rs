use std::{env, fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

mod compile;
mod label;

#[derive(Parser)]
#[command(name = "brier")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Compile {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Label {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        episodes: PathBuf,
        #[arg(long)]
        catalog: PathBuf,
    },
    Ledger {
        #[command(subcommand)]
        command: LedgerCommand,
    },
}

#[derive(Subcommand)]
enum LedgerCommand {
    Init,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Compile { ledger, out } => compile::compile(&ledger, &out)?,
        Command::Label {
            ledger,
            episodes,
            catalog,
        } => label::label(&ledger, &episodes, &catalog)?,
        Command::Ledger {
            command: LedgerCommand::Init,
        } => {
            let data = env::var_os("BRIER_DATA").context("BRIER_DATA must be set")?;
            if data.is_empty() {
                bail!("BRIER_DATA must not be empty");
            }
            if data.to_string_lossy().starts_with("gs://") {
                bail!("SQLite ledger initialization requires a local directory in BRIER_DATA; gs:// is unsupported");
            }
            let directory = PathBuf::from(data);
            fs::create_dir_all(&directory)
                .with_context(|| format!("create BRIER_DATA directory {}", directory.display()))?;
            let path = directory.join("ledger.sqlite");
            brier_core::ledger::init(&path)
                .with_context(|| format!("initialize ledger at {}", path.display()))?;
            println!("Initialized ledger at {}", path.display());
        }
    }
    Ok(())
}
