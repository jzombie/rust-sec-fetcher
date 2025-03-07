use std::collections::HashMap;
use std::hash::Hash;

pub fn invert_multivalue_map<K, V>(map: &HashMap<K, Vec<V>>) -> HashMap<V, Vec<K>>
where
    K: Eq + Hash + Clone,
    V: Eq + Hash + Clone,
{
    let mut inverted_map: HashMap<V, Vec<K>> = HashMap::new();

    for (key, values) in map {
        for value in values {
            inverted_map
                .entry(value.clone())
                .or_insert_with(Vec::new)
                .push(key.clone());
        }
    }

    inverted_map
}
