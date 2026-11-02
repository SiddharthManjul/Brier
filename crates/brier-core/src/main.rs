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
        #[arg(long)]
        heldout_from: Option<String>,
    },
    Label {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        episodes: PathBuf,
        #[arg(long, required_unless_present = "changes", conflicts_with_all = ["changes", "scenarios"])]
        catalog: Option<PathBuf>,
        #[arg(long, requires = "scenarios")]
        changes: Option<PathBuf>,
        #[arg(long, requires = "changes")]
        scenarios: Option<PathBuf>,
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
        #[arg(long, value_enum, default_value = "change")]
        guarantee: brier_core::policy::Guarantee,
    },
    Replay {
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        compiled: PathBuf,
        #[arg(long)]
        #[arg(required_unless_present = "probs", conflicts_with = "probs")]
        scores: Option<PathBuf>,
        #[arg(long, conflicts_with = "scores")]
        probs: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = brier_core::policy::DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, value_delimiter = ',')]
        alphas: Option<Vec<f64>>,
        #[arg(long, value_enum)]
        guarantee: Option<brier_core::policy::Guarantee>,
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
        Command::Compile {
            ledger,
            out,
            heldout_from,
        } => compile::compile_heldout(&ledger, &out, heldout_from.as_deref())?,
        Command::Label {
            ledger,
            episodes,
            catalog,
            changes,
            scenarios,
        } => match catalog {
            Some(catalog) => label::label(&ledger, &episodes, &catalog)?,
            None => label::label_jsonl(&ledger, &episodes, &changes.unwrap(), &scenarios.unwrap())?,
        },
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
            guarantee,
        } => {
            let report = brier_core::decide::decide_guarantee(
                &ledger,
                &compiled,
                &calibration,
                &change,
                alpha,
                guarantee,
            )?;
            println!("{}", serde_json::to_string(&report)?);
        }
        Command::Replay {
            ledger,
            compiled,
            scores,
            out,
            alpha,
            seed,
            probs,
            alphas,
            guarantee,
        } => {
            if let Some(probs) = probs {
                brier_core::replay_report::write_guide3_report(
                    &ledger,
                    &compiled,
                    &probs,
                    &out,
                    &alphas.unwrap_or_else(|| vec![0.05, 0.10]),
                    guarantee.unwrap_or(brier_core::policy::Guarantee::Change),
                    seed,
                )?;
            } else {
                anyhow::ensure!(
                    alphas.is_none() && guarantee.is_none(),
                    "Guide 3 --alphas/--guarantee require --probs with ev/noev scores"
                );
                brier_core::replay_report::write_report(
                    &ledger,
                    &compiled,
                    &scores.unwrap(),
                    &out,
                    alpha,
                    seed,
                )?;
            }
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
