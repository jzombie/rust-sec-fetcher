use sec_fetcher::utils::VecExtensions;
use serde::Serialize;
use std::fs;

#[derive(Debug, Serialize, PartialEq)]
struct Record {
    name: String,
    value: String,
}

#[test]
fn test_head_returns_first_n_elements() {
    let v = vec![1, 2, 3, 4, 5];
    let head = v.head(3);
    assert_eq!(head, &[1, 2, 3]);
}

#[test]
fn test_head_returns_all_when_count_exceeds_length() {
    let v = vec![1, 2, 3];
    let head = v.head(10);
    assert_eq!(head, &[1, 2, 3]);
}

#[test]
fn test_head_empty_vector() {
    let v: Vec<i32> = vec![];
    let head = v.head(3);
    assert!(head.is_empty());
}

#[test]
fn test_head_zero_count() {
    let v = vec![1, 2, 3];
    let head = v.head(0);
    assert!(head.is_empty());
}

#[test]
fn test_tail_returns_last_n_elements() {
    let v = vec![1, 2, 3, 4, 5];
    let tail = v.tail(3);
    assert_eq!(tail, &[3, 4, 5]);
}

#[test]
fn test_tail_returns_all_when_count_exceeds_length() {
    let v = vec![1, 2, 3];
    let tail = v.tail(10);
    assert_eq!(tail, &[1, 2, 3]);
}

#[test]
fn test_tail_empty_vector() {
    let v: Vec<i32> = vec![];
    let tail = v.tail(3);
    assert!(tail.is_empty());
}

#[test]
fn test_tail_zero_count() {
    let v = vec![1, 2, 3];
    let tail = v.tail(0);
    assert!(tail.is_empty());
}

#[test]
fn test_tail_single_element() {
    let v = vec![42];
    let tail = v.tail(1);
    assert_eq!(tail, &[42]);
}

#[test]
fn test_write_to_csv_creates_file() {
    let records = vec![
        Record { name: "Alice".into(), value: "100".into() },
        Record { name: "Bob".into(), value: "200".into() },
    ];

    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_vec_ext_output.csv");
    let path_str = file_path.to_str().unwrap();

    // Remove if left over from a previous failed run
    let _ = fs::remove_file(&file_path);

    records.write_to_csv(path_str).unwrap();

    let contents = fs::read_to_string(&file_path).unwrap();
    assert!(contents.contains("name,value"));
    assert!(contents.contains("Alice,100"));
    assert!(contents.contains("Bob,200"));

    fs::remove_file(&file_path).unwrap();
}

#[test]
fn test_write_to_csv_empty_vector() {
    let v: Vec<Record> = vec![];
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_vec_ext_empty.csv");
    let path_str = file_path.to_str().unwrap();

    let _ = fs::remove_file(&file_path);

    // Should succeed without writing anything
    v.write_to_csv(path_str).unwrap();

    // File won't exist since nothing was written
    assert!(!file_path.exists());
}
