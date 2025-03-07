use indexmap::IndexMap;
use std::hash::Hash;

/// Inverts an `IndexMap<K, Vec<V>>` into `IndexMap<V, Vec<K>>`,
/// preserving the original insertion order.
///
/// # Description
/// This function transforms a mapping where each key (`K`) is associated
/// with multiple values (`Vec<V>`) into a reverse mapping where each value (`V`)
/// now maps to a vector of all keys (`Vec<K>`) that originally contained it.
///
/// The function uses [`IndexMap`](https://docs.rs/indexmap/latest/indexmap/) instead
/// of `HashMap` to maintain insertion order for both keys and values.
///
/// # Type Parameters
/// - `K`: The type of the keys in the original map. Must implement `Eq + Hash + Clone`.
/// - `V`: The type of the values in the original map. Must implement `Eq + Hash + Clone`.
///
/// # Arguments
/// - `map`: A reference to an `IndexMap<K, Vec<V>>` representing the original mapping.
///
/// # Returns
/// - An `IndexMap<V, Vec<K>>` where each unique value from the input map
///   is now a key, and its corresponding value is a vector of original keys.
///
/// # Complexity
/// - **O(N)** where N is the total number of key-value associations.
/// - Lookup and insertion operations are **O(1)** on average due to `IndexMap`.
///
/// # Example
/// ```
/// use indexmap::IndexMap;
/// use sec_fetcher::utils::invert_multivalue_indexmap;
///
/// fn main() {
///     let mut original_map: IndexMap<&str, Vec<&str>> = IndexMap::new();
///     original_map.insert("A", vec!["1", "2"]);
///     original_map.insert("B", vec!["2", "3"]);
///     original_map.insert("C", vec!["1", "3"]);
///
///     let inverted_map = invert_multivalue_indexmap(&original_map);
///
///     assert_eq!(inverted_map.get("1"), Some(&vec!["A", "C"]));
///     assert_eq!(inverted_map.get("2"), Some(&vec!["A", "B"]));
///     assert_eq!(inverted_map.get("3"), Some(&vec!["B", "C"]));
/// }
/// ```
pub fn invert_multivalue_indexmap<K, V>(map: &IndexMap<K, Vec<V>>) -> IndexMap<V, Vec<K>>
where
    K: Eq + Hash + Clone,
    V: Eq + Hash + Clone,
{
    let mut inverted_map: IndexMap<V, Vec<K>> = IndexMap::with_capacity(map.len());

    for (key, values) in map {
        for value in values {
            inverted_map
                .entry(value.clone())
                .or_insert_with(Vec::new) // Explicitly define default behavior
                .push(key.clone());
        }
    }

    inverted_map
}
