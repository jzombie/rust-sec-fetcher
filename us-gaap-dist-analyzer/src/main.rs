use itertools::Itertools;
use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_reduction::Pca;
use ndarray::{Array1, Array2, ArrayBase, Axis, OwnedRepr};
use regex::Regex;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

fn load_column_counts(filename: &str) -> HashMap<String, usize> {
    let file_content =
        std::fs::read_to_string(filename).expect("Failed to read column distribution file");

    let mut column_counts = HashMap::new();

    for line in file_content.lines() {
        if let Some((column, count)) = line.split_once(":") {
            let column = column.trim().to_string();
            if let Ok(count) = count.trim().parse::<usize>() {
                column_counts.insert(column, count);
            }
        }
    }

    column_counts
}

/// Splits column names into words (handles camel case, underscores, numbers).
fn split_column_name(name: &str) -> Vec<String> {
    let mut fixed = name
        .replace('_', " ") // Replace underscores with spaces
        .replace(|c: char| c.is_numeric(), " $0 "); // Add space around numbers

    let mut result = Vec::new();
    let mut current_word = String::new();

    for c in fixed.chars() {
        if c.is_uppercase() && !current_word.is_empty() {
            result.push(current_word);
            current_word = String::new();
        }
        current_word.push(c.to_ascii_lowercase());
    }
    if !current_word.is_empty() {
        result.push(current_word);
    }

    result.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Loads BERT model for embeddings.
fn load_bert_model() -> SentenceEmbeddingsModel {
    SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .create_model()
        .expect("Failed to load BERT model")
}

/// Converts column names into BERT embeddings.
fn get_embeddings(column_names: &[String], model: &SentenceEmbeddingsModel) -> Array2<f32> {
    let processed: Vec<String> = column_names
        .iter()
        .map(|col| split_column_name(col).join(" "))
        .collect();

    let embeddings = model
        .encode(&processed)
        .expect("Failed to generate embeddings");
    let shape = (column_names.len(), embeddings[0].len());

    Array2::from_shape_vec(shape, embeddings.concat()).unwrap()
}

/// Applies PCA to reduce dimensionality to 2D for visualization
fn apply_pca(embeddings: &Array2<f32>, n_components: usize) -> Array2<f32> {
    // Convert embeddings from f32 to f64 (PCA in linfa requires f64)
    let embeddings_f64: Array2<f64> = embeddings.mapv(|v| v as f64);

    // Convert embeddings into a Dataset
    let dataset = Dataset::from(embeddings_f64.clone());

    // Fit PCA
    let pca = Pca::params(n_components).fit(&dataset).expect("PCA failed");

    // PCA transformation requires a dataset, so we wrap records into a dataset
    let transformed_dataset = Dataset::from(embeddings_f64.clone());

    // Perform PCA transformation
    let transformed = pca.transform(transformed_dataset).records().to_owned();

    // Convert back to f32 and return
    transformed.mapv(|v| v as f32)
}

/// Clusters embeddings into groups.
fn cluster_embeddings(embeddings: Array2<f32>, n_clusters: usize) -> Vec<(usize, f32)> {
    let num_samples = embeddings.nrows();

    let dummy_targets: Array1<f64> = Array1::zeros(num_samples);
    let dataset = Dataset::new(embeddings.clone(), dummy_targets);

    let kmeans = KMeans::params(n_clusters)
        .tolerance(1e-3)
        .max_n_iterations(100)
        .fit(&dataset)
        .expect("K-Means clustering failed");

    // Compute distances from cluster centers
    let cluster_assignments = kmeans.predict(&dataset.records);
    let cluster_centers = kmeans.centroids();

    let mut distances = Vec::new();
    for (i, &cluster_id) in cluster_assignments.iter().enumerate() {
        let point = dataset.records.row(i);
        let center = cluster_centers.row(cluster_id);
        let distance = (&point - &center).mapv(|x| x.powi(2)).sum().sqrt(); // Euclidean distance
        distances.push((cluster_id, distance));
    }

    distances
}

fn main() {
    let read_path = "column_distribution_analysis.txt";

    let column_counts = load_column_counts(read_path);

    // print!("Column counts: {:?}", column_counts);
    // print!("{:?}", split_column_name("DerivativeFixedInterestRate"));

    let start = Instant::now();
    println!("Reading column distribution file...");

    let file_content =
        fs::read_to_string(read_path).expect("Failed to read column distribution file");

    let column_names: Vec<String> = file_content
        .lines()
        .map(|line| line.split(':').next().unwrap().trim().to_string())
        .collect();
    println!("Loaded {} column names", column_names.len());

    let model = load_bert_model();
    println!("BERT model loaded. Generating embeddings...");

    let embeddings = get_embeddings(&column_names, &model);
    println!(
        "Generated {} embeddings. Running clustering...",
        embeddings.nrows()
    );

    let clusters = cluster_embeddings(embeddings.clone(), 10);
    println!(
        "Clustering complete in {:.2} seconds. Applying PCA...",
        start.elapsed().as_secs_f32()
    );

    let pca_embeddings = apply_pca(&embeddings, 2); // Reduce to 2D for visualization
    println!("PCA completed. Storing results...");

    // for (col, (cluster_id, distance)) in column_names.iter().zip(clusters.iter()) {
    //     println!("Cluster {} (Distance: {:.5}): {}", cluster_id, distance, col);
    // }

    // Combine column names, cluster IDs, and distances into a vector
    // let mut clustered_data: Vec<(&String, usize, f32)> =
    // column_names.iter().zip(clusters.iter()).map(|(col, (cluster_id, distance))|
    //     (col, *cluster_id, *distance)
    // ).collect();

    // // Sort by cluster ID
    // clustered_data.sort_by_key(|&(_, cluster_id, _)| cluster_id);

    // // Print grouped results
    // for (cluster_id, group) in &clustered_data.iter().group_by(|&(_, id, _)| id) {
    //     println!("\n--- Cluster {} ---", cluster_id);
    //     for (col, _, distance) in group {
    //         println!("(Distance: {:.5}): {}", distance, col);
    //     }
    // }

    // Combine column names, cluster IDs, distances, stock counts, and PCA coordinates
    let mut clustered_data: Vec<(&String, usize, f32, usize, f32, f32)> = column_names
        .iter()
        .zip(clusters.iter())
        .enumerate()
        .map(|(i, (col, (cluster_id, distance)))| {
            let stock_count = column_counts.get(col).copied().unwrap_or(0);
            let pca_x = pca_embeddings[[i, 0]];
            let pca_y = pca_embeddings[[i, 1]];
            (col, *cluster_id, *distance, stock_count, pca_x, pca_y)
        })
        .collect();

    // Sort by cluster ID
    clustered_data.sort_by_key(|&(_, cluster_id, _, _, _, _)| cluster_id);

    // Group by cluster ID and count items per cluster
    let grouped_clusters = clustered_data.iter().group_by(|&(_, id, _, _, _, _)| id);

    for (cluster_id, group) in &grouped_clusters {
        let group_vec: Vec<_> = group.collect();
        let column_count = group_vec.len();

        println!(
            "\n--- Cluster {} ({} columns) ---",
            cluster_id, column_count
        );
        for (col, _, distance, stock_count, pca_x, pca_y) in group_vec {
            println!(
                "(Distance: {:.5}) [Stock Count: {}] [PCA: ({:.3}, {:.3})]: {}",
                distance, stock_count, pca_x, pca_y, col
            );
        }
    }
}
