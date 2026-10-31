use brier_core::report::report;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

static TOKEN: Mutex<()> = Mutex::new(());

struct Fixture {
    directory: PathBuf,
    ledger: PathBuf,
    out: PathBuf,
    connection: Connection,
}

impl Fixture {
    fn new(actions: &[&str], provenance: bool) -> Self {
        let directory = std::env::temp_dir().join(format!("brier-report-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let ledger = directory.join("ledger.sqlite");
        let connection = brier_core::ledger::init(&ledger).unwrap();
        connection.execute("INSERT INTO changes VALUES('c002','gym_pusht/PushT-v0',?1,'1793404800')",
            [json!({"description":"Gain update","families":[{"family":"action_scale","params":{"scale":0.9}}]}).to_string()]).unwrap();
        for (offset, action) in actions.iter().enumerate() {
            let index = offset + 1;
            let scenario = format!("pusht-s{index:02}");
            connection
                .execute(
                    "INSERT INTO scenarios VALUES(?1,'gym_pusht/PushT-v0','{}',0.8)",
                    [&scenario],
                )
                .unwrap();
            let probability = if *action == "skip" { 0.1 } else { 0.8 };
            let mut scores = json!({"regressed":{"yes":probability,"no":1.0-probability}});
            if provenance {
                scores["policy"] = json!({"alpha":0.05,"lambda":0.5,"calibration_changes":19});
                scores["calibration"] = json!({"model_version":format!("example/base@{}","a".repeat(40)),
                    "t1_model_version":format!("t1@{}","b".repeat(64)),
                    "risk_change_ids":(100..119).map(|n|format!("c{n:03}")).collect::<Vec<_>>()});
            }
            connection.execute("INSERT INTO decisions VALUES(?1,'c002',?2,'T1',?3,?4,0.5,?5,1.0,'threshold','1793404801')",
                params![format!("10000000-0000-4000-8000-{index:012}"),scenario,
                    format!("t1@{}","b".repeat(64)),scores.to_string(),action]).unwrap();
        }
        Self {
            out: directory.join("summary.md"),
            directory,
            ledger,
            connection,
        }
    }

    fn run(&self, index: usize, episodes: i64, successes: i64, seconds: f64) {
        self.connection
            .execute(
                "INSERT INTO runs VALUES(?1,'c002',?2,'expensive',?3,?4,0.0,'raw',?5)",
                params![
                    format!("20000000-0000-4000-8000-{index:012}"),
                    format!("pusht-s{index:02}"),
                    episodes,
                    successes,
                    seconds
                ],
            )
            .unwrap();
    }

    fn label(&self, index: usize, regressed: bool) {
        self.connection
            .execute(
                "INSERT INTO labels VALUES('c002',?1,?2,'stall/timeout',1,'action_scale')",
                params![format!("pusht-s{index:02}"), i64::from(regressed)],
            )
            .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

fn restore(previous: Option<std::ffi::OsString>) {
    match previous {
        Some(value) => std::env::set_var("BRIER_INGEST_TOKEN", value),
        None => std::env::remove_var("BRIER_INGEST_TOKEN"),
    }
}

#[test]
fn report_preserves_large_nonzero_change_ids_used_by_policy_ci() {
    let fixture = Fixture::new(&["skip"], true);
    fixture
        .connection
        .execute_batch(
            "DROP TRIGGER changes_no_UPDATE; DROP TRIGGER decisions_no_UPDATE;
         UPDATE changes SET change_id='c123456789012345678' WHERE change_id='c002';
         UPDATE decisions SET change_id='c123456789012345678' WHERE change_id='c002';",
        )
        .unwrap();
    let output = report(
        &fixture.ledger,
        "c123456789012345678",
        &fixture.out,
        None,
        None,
    )
    .unwrap();
    assert_eq!(output.verdict, "pass");
    assert_eq!(
        output.bundle.unwrap()["change"]["id"],
        "c123456789012345678"
    );
}

fn server(status: u16) -> (String, thread::JoinHandle<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/api/ingest", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let handle = thread::spawn(move || {
        let started = Instant::now();
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        started.elapsed() < Duration::from_secs(5),
                        "no local ingest request"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let (offset, length) = loop {
            let mut buffer = [0_u8; 4096];
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(offset) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..offset]);
                assert!(headers.starts_with("POST /api/ingest HTTP/1.1"));
                assert!(headers
                    .to_ascii_lowercase()
                    .contains("authorization: bearer local-test-token"));
                let length = headers
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                    .unwrap()
                    .1
                    .trim()
                    .parse::<usize>()
                    .unwrap();
                break (offset + 4, length);
            }
        };
        while bytes.len() < offset + length {
            let mut buffer = [0_u8; 4096];
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
        }
        let body: Value = serde_json::from_slice(&bytes[offset..offset + length]).unwrap();
        let response = json!({"evaluationId":body["evaluation"]["id"]}).to_string();
        write!(stream,"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nLocation: http://127.0.0.1:1/ignored\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
        body
    });
    (url, handle)
}

#[test]
fn report_passes_completed_selection_without_exposing_skipped_ground_truth() {
    let fixture = Fixture::new(&["expensive", "skip"], true);
    fixture.run(1, 20, 18, 40.0);
    fixture.label(1, false);
    fixture.run(2, 20, 0, 60.0);
    fixture.label(2, true);
    let first = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
    assert_eq!(first.verdict, "pass");
    let bundle = first.bundle.unwrap();
    assert_eq!(bundle["evaluation"]["hoursSavedPct"], 60.0);
    assert_eq!(bundle["evaluation"]["evalsRun"], 1);
    assert_eq!(bundle["pairs"][1]["result"], Value::Null);
    assert_eq!(bundle["pairs"][1]["successAfter"], Value::Null);
    assert!(first.markdown.contains("pusht-s02 | 0.100 | skip | n/a"));
    assert!(!first.markdown.contains("REGRESSED"));
    let second = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
    assert_eq!(second.bundle.unwrap(), bundle);
    assert_eq!(second.markdown, first.markdown);
    assert_eq!(fs::read_to_string(&fixture.out).unwrap(), first.markdown);
}

#[test]
fn report_uses_null_when_live_counterfactual_timings_are_unavailable() {
    let fixture = Fixture::new(&["expensive", "skip"], true);
    fixture.run(1, 20, 18, 40.0);
    fixture.label(1, false);
    let output = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
    assert_eq!(output.verdict, "pass");
    assert!(output.bundle.unwrap()["evaluation"]["hoursSavedPct"].is_null());
    assert!(output
        .markdown
        .contains("Evaluation-count reduction: 50.0%"));
    assert!(output
        .markdown
        .contains("pending unmeasured counterfactual timings"));
    assert!(!output.markdown.contains("50.0% saved"));
}

#[test]
fn report_blocks_confirmed_regressions_and_preserves_real_explanation_evidence() {
    let fixture = Fixture::new(&["expensive"], true);
    fixture.run(1, 20, 5, 40.0);
    fixture.label(1, true);
    let explanation = json!({"failure_mode":"grasp or drop failure","cause_hypothesis":"The cube slipped.",
        "evidence_frames":[0,7],"suggested_check":"Inspect gripper timing.",
        "keyframe_urls":["https://storage.googleapis.com/example/frame.png"],
        "overlay_urls":["https://storage.googleapis.com/example/overlay.png"]});
    fixture
        .connection
        .execute(
            "INSERT INTO explanations VALUES(?1,'Muse Spark',?2,0.01)",
            params![
                "20000000-0000-4000-8000-000000000001",
                explanation.to_string()
            ],
        )
        .unwrap();
    let metrics = fixture.directory.join("report.json");
    fs::write(
        &metrics,
        json!({"evaluation_splits":["test"],"strategies":{}}).to_string(),
    )
    .unwrap();
    let output = report(&fixture.ledger, "c002", &fixture.out, None, Some(&metrics)).unwrap();
    assert_eq!(output.verdict, "block");
    assert!(output
        .markdown
        .contains("Brier: BLOCK (1 regressions confirmed)"));
    assert!(output.markdown.contains("The cube slipped."));
    assert!(output.markdown.contains("[SAM overlay 1]"));
    let bundle = output.bundle.unwrap();
    assert_eq!(bundle["failures"][0]["explanation"], explanation);
    assert_eq!(
        bundle["failures"][0]["keyframeUrls"][0],
        explanation["keyframe_urls"][0]
    );
    assert_eq!(bundle["report"]["id"], bundle["evaluation"]["id"]);
    assert_eq!(bundle["report"]["suite"], "robot");
}

#[test]
fn report_flattens_persisted_episode_explanations_and_keeps_skipped_evidence_private() {
    let fixture = Fixture::new(&["expensive", "skip"], true);
    fixture.run(1, 20, 18, 40.0);
    fixture.label(1, false);
    fixture.run(2, 20, 0, 60.0);
    fixture.label(2, true);
    let episodes: Vec<_> = (1..=2)
        .map(|index| {
            json!({"id":format!("30000000-0000-4000-8000-{index:012}"),
            "seed":index,"failure_mode":"stall/timeout","cause_hypothesis":"The object stopped.",
            "evidence_frames":[0,7],"suggested_check":"Inspect the final frame.",
            "keyframe_urls":[format!("https://storage.googleapis.com/example/{index}/frame.png")],
            "overlay_urls":[format!("https://storage.googleapis.com/example/{index}/overlay.png")]})
        })
        .collect();
    fixture
        .connection
        .execute(
            "INSERT INTO explanations VALUES(?1,'Muse Spark',?2,0.02)",
            params![
                "20000000-0000-4000-8000-000000000001",
                json!({"episodes":episodes}).to_string()
            ],
        )
        .unwrap();
    fixture
        .connection
        .execute(
            "INSERT INTO explanations VALUES(?1,'Muse Spark','not valid JSON',0.01)",
            ["20000000-0000-4000-8000-000000000002"],
        )
        .unwrap();
    let first = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
    let bundle = first.bundle.unwrap();
    assert_eq!(first.verdict, "pass");
    assert_eq!(bundle["failures"].as_array().unwrap().len(), 2);
    for (index, episode) in episodes.iter().enumerate() {
        let failure = &bundle["failures"][index];
        assert_eq!(failure["id"], episode["id"]);
        assert_eq!(failure["explanation"], *episode);
        assert_eq!(failure["pairId"], bundle["pairs"][0]["id"]);
        assert_eq!(failure["keyframeUrls"], episode["keyframe_urls"]);
        assert_eq!(failure["overlayUrls"], episode["overlay_urls"]);
    }
    assert!(bundle["pairs"][1]["result"].is_null());
    assert!(bundle["pairs"][1]["successAfter"].is_null());
    let second = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
    assert_eq!(second.bundle.unwrap(), bundle);
    assert_eq!(second.markdown, first.markdown);
}

#[test]
fn report_rejects_invalid_episode_identity_counts_and_media_before_writing() {
    let episode = json!({"id":"30000000-0000-4000-8000-000000000001",
        "failure_mode":"stall/timeout","keyframe_urls":["https://example.com/frame.png"],
        "overlay_urls":["https://example.com/overlay.png"]});
    for mutation in [
        "id",
        "missing_id",
        "duplicate",
        "count",
        "empty",
        "shape",
        "missing_media",
        "media_type",
        "keyframe",
        "overlay",
        "cross_run_duplicate",
    ] {
        let fixture = Fixture::new(&["expensive", "expensive"], true);
        for index in 1..=2 {
            fixture.run(index, 20, 18, 40.0);
            fixture.label(index, false);
        }
        let mut explanation = json!({"episodes":[episode.clone()]});
        match mutation {
            "id" => {
                explanation["episodes"][0]["id"] = json!("30000000-0000-1000-8000-000000000001")
            }
            "missing_id" => {
                explanation["episodes"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("id");
            }
            "duplicate" => explanation["episodes"] = json!([episode, episode]),
            "count" => explanation["episodes"] = json!([episode, episode, episode]),
            "empty" => explanation["episodes"] = json!([]),
            "shape" => explanation["episodes"] = json!({}),
            "missing_media" => {
                explanation["episodes"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("keyframe_urls");
            }
            "media_type" => {
                explanation["episodes"][0]["overlay_urls"] =
                    json!("https://example.com/overlay.png")
            }
            "keyframe" => {
                explanation["episodes"][0]["keyframe_urls"] = json!(["gs://example/frame.png"])
            }
            "overlay" => {
                explanation["episodes"][0]["overlay_urls"] =
                    json!(["https://user:secret@example.com/overlay.png"])
            }
            "cross_run_duplicate" => {
                fixture
                    .connection
                    .execute(
                        "INSERT INTO explanations VALUES(?1,'Muse Spark',?2,0.01)",
                        params![
                            "20000000-0000-4000-8000-000000000002",
                            explanation.to_string()
                        ],
                    )
                    .unwrap();
            }
            _ => unreachable!(),
        }
        fixture
            .connection
            .execute(
                "INSERT INTO explanations VALUES(?1,'Muse Spark',?2,0.01)",
                params![
                    "20000000-0000-4000-8000-000000000001",
                    explanation.to_string()
                ],
            )
            .unwrap();
        assert!(
            report(&fixture.ledger, "c002", &fixture.out, None, None).is_err(),
            "{mutation}"
        );
        assert!(!fixture.out.exists(), "{mutation}");
    }
}

#[test]
fn report_stays_pending_for_missing_partial_or_unlabelled_selected_runs() {
    for state in ["missing", "partial", "unlabelled"] {
        let fixture = Fixture::new(&["expensive"], true);
        if state != "missing" {
            fixture.run(1, if state == "partial" { 19 } else { 20 }, 5, 40.0);
        }
        if state != "unlabelled" {
            fixture.label(1, true);
        }
        let output = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
        assert_eq!(output.verdict, "pending");
        assert!(output.bundle.is_none());
        assert!(output.markdown.contains("Brier: PENDING"));
        assert!(!output.markdown.contains("REGRESSED"));
    }
}

#[test]
fn report_preserves_a_confirmed_block_while_other_selected_results_are_pending() {
    let fixture = Fixture::new(&["expensive", "expensive"], true);
    fixture.run(1, 20, 5, 40.0);
    fixture.label(1, true);
    let output = report(&fixture.ledger, "c002", &fixture.out, None, None).unwrap();
    assert_eq!(output.verdict, "block");
    assert!(output.bundle.is_none());
    assert!(output.markdown.contains("1 pending"));
}

#[test]
fn report_rejects_legacy_policy_provenance_and_input_replacement() {
    let fixture = Fixture::new(&["skip"], false);
    let error = report(&fixture.ledger, "c002", &fixture.out, None, None)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("policy provenance"));
    assert!(report(&fixture.ledger, "c002", &fixture.ledger, None, None).is_err());
    assert_eq!(
        fixture
            .connection
            .query_row("SELECT count(*) FROM changes", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn report_missing_token_performs_no_http_request() {
    let _guard = TOKEN.lock().unwrap();
    let fixture = Fixture::new(&["skip"], true);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/api/ingest", listener.local_addr().unwrap());
    let previous = std::env::var_os("BRIER_INGEST_TOKEN");
    std::env::remove_var("BRIER_INGEST_TOKEN");
    let result = report(&fixture.ledger, "c002", &fixture.out, Some(&endpoint), None);
    restore(previous);
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("BRIER_INGEST_TOKEN"));
    assert_eq!(
        listener.accept().err().unwrap().kind(),
        std::io::ErrorKind::WouldBlock
    );
    assert!(fixture.out.exists());
    assert!(!fs::read_to_string(&fixture.out)
        .unwrap()
        .contains("View evaluation"));
    let pending = Fixture::new(&["expensive"], true);
    let result = report(&pending.ledger, "c002", &pending.out, Some(&endpoint), None);
    assert!(result.err().unwrap().to_string().contains("pending"));
    assert_eq!(
        listener.accept().err().unwrap().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn report_push_sends_one_valid_web_bundle_and_derives_the_run_link() {
    let _guard = TOKEN.lock().unwrap();
    let fixture = Fixture::new(&["skip"], true);
    let (endpoint, handle) = server(200);
    let previous = std::env::var_os("BRIER_INGEST_TOKEN");
    std::env::set_var("BRIER_INGEST_TOKEN", "local-test-token");
    let result = report(&fixture.ledger, "c002", &fixture.out, Some(&endpoint), None);
    restore(previous);
    let output = result.unwrap();
    let received = handle.join().unwrap();
    assert_eq!(output.bundle.as_ref().unwrap(), &received);
    assert!(output
        .markdown
        .contains(&endpoint.replace("/api/ingest", "/runs/10000000-0000-4000-8000-000000000001")));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = "import {validateIngest} from './apps/web/lib/ingest.ts'; let body=''; for await(const part of process.stdin) body+=part; process.stdout.write(JSON.stringify(validateIngest(JSON.parse(body))));";
    let mut node = Command::new("node")
        .args([
            "--experimental-strip-types",
            "--input-type=module",
            "-e",
            script,
        ])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    node.stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&received).unwrap())
        .unwrap();
    let checked = node.wait_with_output().unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&checked.stdout).unwrap(),
        received
    );
}

#[test]
fn report_push_does_not_follow_redirects() {
    let _guard = TOKEN.lock().unwrap();
    let fixture = Fixture::new(&["skip"], true);
    let (endpoint, handle) = server(302);
    let previous = std::env::var_os("BRIER_INGEST_TOKEN");
    std::env::set_var("BRIER_INGEST_TOKEN", "local-test-token");
    let result = report(&fixture.ledger, "c002", &fixture.out, Some(&endpoint), None);
    restore(previous);
    assert!(result.err().unwrap().to_string().contains("302"));
    handle.join().unwrap();
}

#[test]
fn report_cli_maps_only_published_gcs_media_and_keeps_local_evidence_pending() {
    let fixture = Fixture::new(&["expensive"], true);
    fixture.run(1, 20, 19, 40.0);
    fixture.label(1, true);
    fixture.connection.execute("INSERT INTO explanations VALUES(?1,'Muse Spark',?2,NULL)",params!["20000000-0000-4000-8000-000000000001",json!({"episodes":[{
        "id":"30000000-0000-4000-8000-000000000001","failure_mode":"stall/timeout","cause_hypothesis":"action_latency\nThe object stopped.","evidence_frames":[0],"suggested_check":"Inspect the frame.",
        "keyframe_urls":[],"overlay_urls":[],"local_keyframes":["public/keyframes/push t/frame-é%20.png"],"local_overlays":["public/overlays/episode/overlay 0.png"]}]}).to_string()]).unwrap();
    let run = |root: &str| {
        let result = Command::new(env!("CARGO_BIN_EXE_brier"))
            .args(["report", "--ledger"])
            .arg(&fixture.ledger)
            .args(["--change", "c002", "--out"])
            .arg(&fixture.out)
            .env("BRIER_DATA", root)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        let output: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(output["verdict"], "block");
        assert_eq!(
            output["markdown"],
            fs::read_to_string(&fixture.out).unwrap()
        );
        output
    };
    let cloud = run("gs://brier-test/ci outputs");
    let failure = &cloud["bundle"]["failures"][0];
    assert_eq!(failure["keyframeUrls"],json!(["https://storage.googleapis.com/brier-test/ci%20outputs/public/keyframes/push%20t/frame-%C3%A9%2520.png"]));
    assert_eq!(failure["overlayUrls"],json!(["https://storage.googleapis.com/brier-test/ci%20outputs/public/overlays/episode/overlay%200.png"]));
    assert!(!cloud["markdown"]
        .as_str()
        .unwrap()
        .contains("Evidence publication: pending"));
    let local = run(fixture.directory.to_str().unwrap());
    assert_eq!(local["bundle"]["failures"][0]["keyframeUrls"], json!([]));
    assert_eq!(local["bundle"]["failures"][0]["overlayUrls"], json!([]));
    assert!(local["markdown"]
        .as_str()
        .unwrap()
        .contains("Evidence publication: pending"));
}

#[test]
fn report_keeps_markdown_when_push_configuration_is_invalid() {
    let fixture = Fixture::new(&["skip"], true);
    assert!(report(
        &fixture.ledger,
        "c002",
        &fixture.out,
        Some("http://example.com/api/ingest"),
        None
    )
    .is_err());
    assert!(fs::read_to_string(&fixture.out)
        .unwrap()
        .contains("Brier: PASS"));
}
