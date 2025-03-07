use indexmap::IndexMap;
use std::hash::Hash;

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
