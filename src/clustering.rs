use anyhow::{anyhow, Result};
use ndarray::Array2;
use petal_clustering::{Fit as PetalFit, HDbscan};
use petal_neighbors::distance::Euclidean;
use std::collections::HashMap;
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::{GaussianMixtureModel, GmmValidParams, KMeans};
use rand_xoshiro::Xoshiro256Plus;
use rand::SeedableRng;

/// Result of a clustering operation
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Mapping of cluster IDs to the indices of data points in that cluster
    pub clusters: HashMap<usize, Vec<usize>>,
    /// Indices of data points considered as outliers
    pub outliers: Vec<usize>,
    /// Flattened representation of cluster assignments (index = data point, value = cluster ID)
    pub assignments: Vec<usize>,
}

/// Performs HDBSCAN clustering on a dataset
///
/// # Arguments
/// * `data` - A 2D array of data points to cluster
/// * `min_cluster_size` - Minimum number of points to form a cluster
/// * `min_samples` - Minimum number of neighbors required for a point to be considered a core point
/// * `epsilon` - Distance threshold for connecting points (default: 0.0001)
/// * `alpha` - Factor for determining cluster prominence (default: 1.0)
///
/// # Returns
/// * `Result<ClusteringResult>` - The clustering result or error
pub fn hdbscan_clustering(
    data: &[Vec<f64>],
    min_cluster_size: usize,
    min_samples: usize,
    epsilon: Option<f64>,
    alpha: Option<f64>,
) -> Result<ClusteringResult> {
    // Convert data to ndarray format
    let nrows = data.len();
    if nrows == 0 {
        return Err(anyhow!("Empty input data"));
    }
    
    let ncols = data[0].len();
    let flat_data: Vec<f64> = data.iter().flat_map(|v| v.iter().cloned()).collect();
    
    let data_array = Array2::from_shape_vec((nrows, ncols), flat_data)
        .map_err(|e| anyhow!("Failed to reshape data: {}", e))?;
    
    // Create HDBSCAN algorithm with parameters
    let mut hdbscan = HDbscan {
        eps: epsilon.unwrap_or(0.0001),
        alpha: alpha.unwrap_or(1.0),
        min_samples,
        min_cluster_size,
        metric: Euclidean::default(),
        boruvka: true,
    };
    
    // Perform clustering
    let (clusters, outliers) = PetalFit::fit(&mut hdbscan, &data_array);
    
    // Create cluster assignments vector (0 is reserved for outliers)
    let mut assignments = vec![0; nrows];
    for (cluster_id, indices) in clusters.iter() {
        for &idx in indices {
            assignments[idx] = *cluster_id;
        }
    }
    
    Ok(ClusteringResult {
        clusters,
        outliers,
        assignments,
    })
}

/// Performs GMM (Gaussian Mixture Model) clustering on a dataset
///
/// # Arguments
/// * `data` - A 2D array of data points to cluster
/// * `n_clusters` - Number of clusters to create
/// * `n_runs` - Number of runs to perform (default: 10)
/// * `tolerance` - Convergence tolerance (default: 1e-4)
/// * `seed` - Random seed for reproducibility (default: 42)
///
/// # Returns
/// * `Result<ClusteringResult>` - The clustering result or error
pub fn gmm_clustering(
    data: &[Vec<f64>],
    n_clusters: usize,
    n_runs: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
) -> Result<ClusteringResult> {
    // Check for empty data
    let nrows = data.len();
    if nrows == 0 {
        return Err(anyhow!("Empty input data"));
    }
    
    // Convert data to ndarray format for linfa
    let ncols = data[0].len();
    let flat_data: Vec<f64> = data.iter().flat_map(|v| v.iter().cloned()).collect();
    
    let data_array = Array2::from_shape_vec((nrows, ncols), flat_data)
        .map_err(|e| anyhow!("Failed to reshape data: {}", e))?;
    
    // Create dataset for GMM
    let dataset = DatasetBase::from(data_array);
    
    // Initialize random number generator
    let rng = Xoshiro256Plus::seed_from_u64(seed.unwrap_or(42));
    
    // Configure and run GMM
    let gmm = GaussianMixtureModel::params(n_clusters)
        .n_runs(n_runs.unwrap_or(10) as u64)
        .tolerance(tolerance.unwrap_or(1e-4))
        .with_rng(rng)
        .fit(&dataset)
        .map_err(|e| anyhow!("GMM fitting failed: {}", e))?;
    
    // Get cluster assignments
    let clustered_data = gmm.predict(dataset);
    let targets = clustered_data.targets();
    
    // Convert to the ClusteringResult format
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut assignments = vec![0; nrows];
    
    for (idx, &cluster_id) in targets.iter().enumerate() {
        // Store assignment
        let cluster_id_usize = cluster_id as usize;
        assignments[idx] = cluster_id_usize;
        
        // Add to clusters map
        clusters.entry(cluster_id_usize)
            .or_insert_with(Vec::new)
            .push(idx);
    }
    
    // GMM assigns all points to clusters, so there are no outliers
    let outliers = Vec::new();
    
    Ok(ClusteringResult {
        clusters,
        outliers,
        assignments,
    })
}

/// Performs K-means clustering on a dataset
///
/// # Arguments
/// * `data` - A 2D array of data points to cluster
/// * `n_clusters` - Number of clusters to create
/// * `max_iterations` - Maximum number of iterations (default: 100)
/// * `tolerance` - Convergence tolerance (default: 1e-4)
/// * `seed` - Random seed for reproducibility (default: 42)
///
/// # Returns
/// * `Result<ClusteringResult>` - The clustering result or error
pub fn kmeans_clustering(
    data: &[Vec<f64>],
    n_clusters: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
) -> Result<ClusteringResult> {
    // Check for empty data
    let nrows = data.len();
    if nrows == 0 {
        return Err(anyhow!("Empty input data"));
    }
    
    // Convert data to ndarray format for linfa
    let ncols = data[0].len();
    let flat_data: Vec<f64> = data.iter().flat_map(|v| v.iter().cloned()).collect();
    
    let data_array = Array2::from_shape_vec((nrows, ncols), flat_data)
        .map_err(|e| anyhow!("Failed to reshape data: {}", e))?;
    
    // Create dataset for KMeans
    let dataset = DatasetBase::from(data_array);
    
    // Initialize random number generator
    let rng = Xoshiro256Plus::seed_from_u64(seed.unwrap_or(42));
    
    // Configure and run KMeans
    let kmeans = KMeans::params_with_rng(n_clusters, rng)
        .max_n_iterations(max_iterations.unwrap_or(100) as u64)
        .tolerance(tolerance.unwrap_or(1e-4))
        .fit(&dataset)
        .map_err(|e| anyhow!("KMeans fitting failed: {}", e))?;
    
    // Get cluster assignments
    let clustered_data = kmeans.predict(dataset);
    let targets = clustered_data.targets();
    
    // Convert to the ClusteringResult format
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut assignments = vec![0; nrows];
    
    for (idx, &cluster_id) in targets.iter().enumerate() {
        // Store assignment
        let cluster_id_usize = cluster_id as usize;
        assignments[idx] = cluster_id_usize;
        
        // Add to clusters map
        clusters.entry(cluster_id_usize)
            .or_insert_with(Vec::new)
            .push(idx);
    }
    
    // KMeans assigns all points to clusters, so there are no outliers
    let outliers = Vec::new();
    
    Ok(ClusteringResult {
        clusters,
        outliers,
        assignments,
    })
}

/// Group items by their cluster assignment
///
/// # Arguments
/// * `cluster_assignments` - Vector of cluster assignments (index = data point, value = cluster ID)
/// * `items` - Vector of items to group by cluster assignment
///
/// # Returns
/// * `HashMap<usize, Vec<T>>` - Mapping of cluster IDs to vectors of items
pub fn group_by_cluster<T: Clone>(
    cluster_assignments: &[usize],
    items: &[T],
) -> HashMap<usize, Vec<T>> {
    if cluster_assignments.len() != items.len() {
        return HashMap::new();
    }
    
    let mut result: HashMap<usize, Vec<T>> = HashMap::new();
    
    for (idx, &cluster) in cluster_assignments.iter().enumerate() {
        result.entry(cluster)
            .or_insert_with(Vec::new)
            .push(items[idx].clone());
    }
    
    result
} 