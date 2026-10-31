use std::{env, fs, path::PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use brier_core::compile;
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
    Card {
        #[arg(long)]
        before: PathBuf,
        #[arg(long)]
        after: PathBuf,
        #[arg(long)]
        change: String,
    },
    Decide {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        compiled: PathBuf,
        #[arg(long)]
        calibration: PathBuf,
        #[arg(long)]
        change: String,
        #[arg(long, default_value_t = brier_core::policy::DEFAULT_ALPHA)]
        alpha: f64,
    },
    Replay {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        compiled: PathBuf,
        #[arg(long)]
        scores: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = brier_core::policy::DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    Report {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        change: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        push: Option<String>,
        #[arg(long)]
        metrics: Option<PathBuf>,
    },
    Explain {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        run: String,
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
        Command::Card {
            before,
            after,
            change,
        } => {
            let card = brier_core::card::card(&before, &after, &change)?;
            println!("{}", serde_json::to_string(&card)?);
        }
        Command::Decide {
            ledger,
            compiled,
            calibration,
            change,
            alpha,
        } => {
            let report =
                brier_core::decide::decide(&ledger, &compiled, &calibration, &change, alpha)?;
            println!("{}", serde_json::to_string(&report)?);
        }
        Command::Replay {
            ledger,
            compiled,
            scores,
            out,
            alpha,
            seed,
        } => {
            brier_core::replay_report::write_report(
                &ledger, &compiled, &scores, &out, alpha, seed,
            )?;
        }
        Command::Report {
            ledger,
            change,
            out,
            push,
            metrics,
        } => {
            let report = brier_core::report::report(
                &ledger,
                &change,
                &out,
                push.as_deref(),
                metrics.as_deref(),
            )?;
            println!("{}", serde_json::to_string(&report)?);
            if report.verdict != "pass" {
                bail!("Brier: {}", report.verdict.to_uppercase());
            }
        }
        Command::Explain { ledger, run } => {
            let report = brier_core::explain::explain(&ledger, &run)?;
            println!("{}", serde_json::to_string(&report)?);
        }
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
