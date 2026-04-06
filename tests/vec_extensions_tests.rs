/// Unit tests for [`sec_fetcher::utils::VecExtensions`].
use sec_fetcher::utils::VecExtensions;

// ── head ──────────────────────────────────────────────────────────────────────

#[test]
fn head_returns_first_n_elements() {
    let v = vec![1, 2, 3, 4, 5];
    assert_eq!(v.head(3), &[1, 2, 3]);
}

#[test]
fn head_clamps_to_len_when_n_exceeds_len() {
    let v = vec![1, 2, 3];
    assert_eq!(v.head(10), &[1, 2, 3]);
}

#[test]
fn head_zero_returns_empty_slice() {
    let v = vec![1, 2, 3];
    assert_eq!(v.head(0), &[] as &[i32]);
}

#[test]
fn head_on_empty_vec_returns_empty_slice() {
    let v: Vec<i32> = vec![];
    assert_eq!(v.head(5), &[] as &[i32]);
}

#[test]
fn head_exact_length() {
    let v = vec![10, 20, 30];
    assert_eq!(v.head(3), &[10, 20, 30]);
}

// ── tail ──────────────────────────────────────────────────────────────────────

#[test]
fn tail_returns_last_n_elements() {
    let v = vec![1, 2, 3, 4, 5];
    assert_eq!(v.tail(3), &[3, 4, 5]);
}

#[test]
fn tail_clamps_to_len_when_n_exceeds_len() {
    let v = vec![1, 2, 3];
    assert_eq!(v.tail(10), &[1, 2, 3]);
}

#[test]
fn tail_zero_returns_empty_slice() {
    let v = vec![1, 2, 3];
    assert_eq!(v.tail(0), &[] as &[i32]);
}

#[test]
fn tail_on_empty_vec_returns_empty_slice() {
    let v: Vec<i32> = vec![];
    assert_eq!(v.tail(5), &[] as &[i32]);
}

#[test]
fn tail_exact_length() {
    let v = vec![10, 20, 30];
    assert_eq!(v.tail(3), &[10, 20, 30]);
}

#[test]
fn tail_one_element() {
    let v = vec![1, 2, 3, 4, 5];
    assert_eq!(v.tail(1), &[5]);
}
