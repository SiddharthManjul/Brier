use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

fn temporary_directory() -> PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "brier-cli-{}-{nonce}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ))
}

fn init_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_brier"));
    command.args(["ledger", "init"]).env_remove("BRIER_DATA");
    command
}

#[test]
fn initializes_local_ledger_with_all_tables() {
    let directory = temporary_directory().join("nested");
    let output = init_command()
        .env("BRIER_DATA", &directory)
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let path = directory.join("ledger.sqlite");
    assert!(path.is_file());
    let connection = rusqlite::Connection::open(&path).unwrap();
    let tables: Vec<String> = connection
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        tables,
        [
            "changes",
            "decisions",
            "explanations",
            "labels",
            "runs",
            "scenarios"
        ]
    );
    drop(connection);
    fs::remove_dir_all(directory.parent().unwrap()).unwrap();
}

#[test]
fn rejects_missing_and_empty_data_directory() {
    let missing = init_command().output().unwrap();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("BRIER_DATA must be set"));
    let empty = init_command().env("BRIER_DATA", "").output().unwrap();
    assert!(!empty.status.success());
    assert!(String::from_utf8_lossy(&empty.stderr).contains("BRIER_DATA must not be empty"));
}

#[test]
fn rejects_gcs_for_local_sqlite() {
    let output = init_command()
        .env("BRIER_DATA", "gs://test-bucket/brier")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("local directory"));
    assert!(error.contains("gs:// is unsupported"));
}

#[test]
fn existing_ledger_fails_without_changing_rows() {
    let directory = temporary_directory();
    assert!(init_command()
        .env("BRIER_DATA", &directory)
        .status()
        .unwrap()
        .success());
    let path = directory.join("ledger.sqlite");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute("INSERT INTO changes (change_id) VALUES (?1)", ["c001"])
        .unwrap();
    let output = init_command()
        .env("BRIER_DATA", &directory)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let id: String = connection
        .query_row("SELECT change_id FROM changes", [], |row| row.get(0))
        .unwrap();
    assert_eq!(id, "c001");
    drop(connection);
    fs::remove_dir_all(directory).unwrap();
}
