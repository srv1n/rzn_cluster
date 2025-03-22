use ndarray::Array2;

/// Convert a 2D vector to ndarray Array2<f64>
///
/// # Arguments
/// * `data` - The 2D vector to convert
///
/// # Returns
/// * `Array2<f64>` - The resulting 2D array
pub fn vec_to_array2(data: &[Vec<f64>]) -> Array2<f64> {
    if data.is_empty() {
        return Array2::from_shape_vec((0, 0), vec![]).unwrap();
    }
    
    let nrows = data.len();
    let ncols = data[0].len();
    let flat_data: Vec<f64> = data.iter().flat_map(|v| v.iter().cloned()).collect();
    
    Array2::from_shape_vec((nrows, ncols), flat_data).unwrap()
}

/// Compute Euclidean distance between two vectors
///
/// # Arguments
/// * `v1` - First vector
/// * `v2` - Second vector
///
/// # Returns
/// * `f64` - Euclidean distance
pub fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.len() != v2.len() {
        panic!("Vectors must have the same length");
    }
    
    v1.iter()
        .zip(v2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute cosine similarity between two vectors
///
/// # Arguments
/// * `v1` - First vector
/// * `v2` - Second vector
///
/// # Returns
/// * `f64` - Cosine similarity (-1 to 1, where 1 means identical direction)
pub fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.len() != v2.len() {
        panic!("Vectors must have the same length");
    }
    
    let dot_product = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum::<f64>();
    
    let mag1 = v1.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
    let mag2 = v2.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
    
    dot_product / (mag1 * mag2)
} 