use anyhow::Result;
use std::collections::HashSet;
use rand::seq::SliceRandom;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use hnsw_rs::prelude::*;
use annembed::prelude::*;
use annembed::fromhnsw::kgraph::{kgraph_from_hnsw_all, KGraph};

/// Result structure returned by dimensionality reduction functions
#[derive(Clone, Debug)]
pub struct EmbeddingResult {
    /// The reduced-dimension embeddings
    pub embeddings: Vec<Vec<f64>>,
    /// Original indices of the data points (used when sampling)
    pub original_indices: Vec<usize>,
}

/// Performs dimensionality reduction on input data using HNSW and Annembed
///
/// # Arguments
/// * `input_data` - A slice of vectors representing the high-dimensional data points
/// * `output_dim` - The target dimensionality to reduce to
/// * `sample_size` - Optional parameter to use only a subset of data for faster computation
///
/// # Returns
/// * `Result<EmbeddingResult, Box<dyn std::error::Error>>` - The reduced embeddings and original indices
pub fn perform_dimension_reduction(
    input_data: &[Vec<f64>],
    output_dim: usize,
    sample_size: Option<usize>,
) -> Result<EmbeddingResult, Box<dyn std::error::Error>> {
    let (data_to_use, original_indices) = if let Some(size) = sample_size {
        let size = std::cmp::min(size, input_data.len());
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let mut indices: Vec<usize> = (0..input_data.len()).collect();
        indices.shuffle(&mut rng);
        let sample_indices = indices[0..size].to_vec();
        let unique_indices: HashSet<usize> = sample_indices.iter().cloned().collect();
        let mut sorted_indices: Vec<usize> = unique_indices.into_iter().collect();
        sorted_indices.sort();
        
        (
            sorted_indices
                .iter()
                .map(|&idx| input_data[idx].clone())
                .collect::<Vec<Vec<f64>>>(),
            sorted_indices,
        )
    } else {
        (
            input_data.to_vec(),
            (0..input_data.len()).collect(),
        )
    };

    // Create HNSW index
    let ef_c = 50;
    let max_nb_connection = 70;
    let nb_layer = 16.min((data_to_use.len() as f64).ln().trunc() as usize);
    
    let hnsw = Hnsw::<f64, DistL2>::new(
        max_nb_connection,
        data_to_use.len(),
        nb_layer,
        ef_c,
        DistL2 {},
    );

    // Insert data into HNSW
    let data_with_id: Vec<(&Vec<f64>, usize)> =
        data_to_use.iter().zip(0..data_to_use.len()).collect();
    hnsw.parallel_insert(&data_with_id);

    // Create KGraph
    let knbn = 6;
    let kgraph: KGraph<f64> = kgraph_from_hnsw_all(&hnsw, knbn)
        .map_err(|e| anyhow::anyhow!("Failed to create KGraph: {}", e))?;

    // Set up Embedder
    let mut embed_params = EmbedderParams::default();
    embed_params.nb_grad_batch = 30;
    embed_params.scale_rho = 1.;
    embed_params.beta = 1.;
    embed_params.grad_step = 1.;
    embed_params.nb_sampling_by_edge = 10;
    embed_params.dmap_init = true;
    embed_params.asked_dim = output_dim;
    
    let mut embedder = Embedder::new(&kgraph, embed_params);
    embedder.embed()
        .map_err(|e| anyhow::anyhow!("Failed to embed: {}", e))?;

    // Get embedded data
    let embedded_data = embedder.get_embedded_reindexed();
    let embeddings: Vec<Vec<f64>> = embedded_data.outer_iter().map(|row| row.to_vec()).collect();

    Ok(EmbeddingResult {
        embeddings,
        original_indices,
    })
} 