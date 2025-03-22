# RZN Cluster Library

A lightweight Rust library for dimensionality reduction and clustering of high-dimensional data. RZN Cluster provides simple, efficient implementations of popular algorithms for data scientists and machine learning enthusiasts.

## ✨ Features

- **Dimensionality Reduction**: Reduce high-dimensional data to lower dimensions using UMAP-inspired techniques
- **Multiple Clustering Algorithms**:
  - **HDBSCAN**: Density-based clustering that can find clusters of varying shapes and sizes
  - **GMM**: Gaussian Mixture Models for probabilistic clustering
  - **K-means**: Classic centroid-based clustering for well-separated, roughly spherical clusters
- **Simple Data Structures**: Works with standard Rust vectors and arrays for easy integration
- **Utility Functions**: Helper functions for data manipulation and analysis

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rzn_cluster = "0.1.0"
```

For macOS acceleration:

```toml
[dependencies]
rzn_cluster = { version = "0.1.0", features = ["macos-accelerate"] }
```

## 🚀 Example Usage

### Dimensionality Reduction

```rust
use rzn_cluster::dimensionality_reduction::perform_dimension_reduction;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Some high-dimensional data
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1.1, 2.1, 3.1, 4.1, 5.1],
        // ... more data points
    ];
    
    // Reduce to 2 dimensions
    let result = perform_dimension_reduction(&data, 2, None)?;
    
    // Now result.embeddings contains the reduced data
    for embedding in result.embeddings.iter() {
        println!("Reduced: {:?}", embedding);
    }
    
    Ok(())
}
```

### HDBSCAN Clustering

```rust
use rzn_cluster::clustering::hdbscan_clustering;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Some data points
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![5.0, 5.0],
        vec![5.1, 5.1],
        // ... more data points
    ];
    
    // Perform HDBSCAN clustering
    let min_cluster_size = 2;
    let min_samples = 1;
    let result = hdbscan_clustering(&data, min_cluster_size, min_samples, None, None)?;
    
    // Print cluster assignments
    println!("Clusters: {:?}", result.clusters);
    println!("Outliers: {:?}", result.outliers);
    
    Ok(())
}
```

### GMM Clustering

```rust
use rzn_cluster::clustering::gmm_clustering;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Some data points
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![5.0, 5.0],
        vec![5.1, 5.1],
        // ... more data points
    ];
    
    // Perform GMM clustering with 2 clusters
    let n_clusters = 2;
    let result = gmm_clustering(&data, n_clusters, None, None, None)?;
    
    // Print cluster assignments
    println!("Clusters: {:?}", result.clusters);
    
    Ok(())
}
```

### K-means Clustering

```rust
use rzn_cluster::clustering::kmeans_clustering;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Some data points
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![5.0, 5.0],
        vec![5.1, 5.1],
        // ... more data points
    ];
    
    // Perform K-means clustering with 2 clusters
    let n_clusters = 2;
    let result = kmeans_clustering(&data, n_clusters, None, None, None)?;
    
    // Print cluster assignments
    println!("Clusters: {:?}", result.clusters);
    
    Ok(())
}
```

## 📊 Examples

See the `examples` directory for more complete examples with visualizations:

- `hdbscan_demo.rs`: Demonstrates HDBSCAN clustering
- `gmm_demo.rs`: Shows Gaussian Mixture Model clustering
- `kmeans_demo.rs`: Illustrates K-means clustering
- `umap_demo.rs`: Demonstrates dimensionality reduction on high-dimensional data
- `lancedb_demo.rs`: Integration with LanceDB for vector database functionality

Run examples with:

```bash
# Clustering examples
cargo run --example hdbscan_demo
cargo run --example gmm_demo
cargo run --example kmeans_demo

# Dimensionality reduction
cargo run --example umap_demo

# Vector database integration
cargo run --example lancedb_demo
```

Each example generates a visualization of the clustering results, allowing you to see how different algorithms perform on the same data.

## 🧩 Design Philosophy

RZN Cluster is designed to be:

- **Simple**: Easy to use with minimal configuration
- **Flexible**: Works with different data types and formats
- **Lightweight**: Core functionality has minimal dependencies
- **Educational**: Clear implementations that help users understand the algorithms

The library deliberately focuses on core dimensionality reduction and clustering functionality, making it ideal for both learning and production use.

## 🛠️ Features

- `macos-accelerate`: Enables BLAS acceleration on macOS for improved performance

## 📝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📄 License

MIT 