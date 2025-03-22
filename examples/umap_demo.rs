// The example generates synthetic high-dimensional data (50 dimensions) with 3 clusters:
// 3 clusters of 50 points each, centered at different locations
// 20 additional random noise points
// It uses the perform_dimension_reduction function from the rzn_cluster library to reduce the dimensionality to 2D using HNSW (Hierarchical Navigable Small World) graph and embedding.
// It visualizes the reduced 2D data, coloring points based on their original clusters.
// The implementation in src/dimensionality_reduction.rs uses:
// The hnsw_rs library to create a proximity graph of the high-dimensional data
// The annembed library for the actual dimensionality reduction
// This appears to be a UMAP-like algorithm since it's creating a graph representation of the data and then finding a low-dimensional embedding that preserves the graph structure.

use anyhow::Result;
use plotters::prelude::*;
use rand::distributions::Distribution;
use rand_distr::Normal;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rzn_cluster::dimensionality_reduction::perform_dimension_reduction;

fn main() -> Result<()> {
    // Set up a random number generator with a fixed seed for reproducibility
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    
    // Generate data in high dimensions
    let n_points_per_cluster = 50;
    let n_dimensions = 50; // High-dimensional data
    let mut high_dim_data = Vec::new();
    
    // Generate a dataset with 3 clusters in high-dimensional space
    let centers = [
        vec![1.0; n_dimensions],
        vec![-1.0; n_dimensions],
        vec![0.0; n_dimensions],
    ];
    
    for center in centers.iter() {
        for _ in 0..n_points_per_cluster {
            let mut point = vec![0.0; n_dimensions];
            // Add some noise to each dimension around the center
            for (i, &center_val) in center.iter().enumerate() {
                let normal_dist = Normal::new(center_val, 0.1).unwrap();
                point[i] = normal_dist.sample(&mut rng);
            }
            high_dim_data.push(point);
        }
    }
    
    // Add some random noise points
    for _ in 0..20 {
        let mut point = vec![0.0; n_dimensions];
        // Generate random points
        for i in 0..n_dimensions {
            let normal_dist = Normal::new(0.0, 5.0).unwrap();
            point[i] = normal_dist.sample(&mut rng);
        }
        high_dim_data.push(point);
    }
    
    println!("Generated {} high-dimensional data points with {} dimensions", 
             high_dim_data.len(), n_dimensions);
    
    // Perform dimensionality reduction to 2D
    println!("Performing dimensionality reduction to 2D using HNSW-based embedding...");
    let output_dim = 2;
    let result = perform_dimension_reduction(&high_dim_data, output_dim, None).unwrap();
    
    println!("Dimensionality reduction complete");
    println!("Original dimensions: {}", n_dimensions);
    println!("Reduced dimensions: {}", output_dim);
    println!("Number of points: {}", result.embeddings.len());
    
    // Plot the 2D embeddings
    let root = BitMapBackend::new("hnsw_embedding_demo.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let embeddings = &result.embeddings;
    let x_values: Vec<f64> = embeddings.iter().map(|v| v[0]).collect();
    let y_values: Vec<f64> = embeddings.iter().map(|v| v[1]).collect();
    
    let min_x = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Add some padding to the plot
    let x_range = (min_x - 1.0)..(max_x + 1.0);
    let y_range = (min_y - 1.0)..(max_y + 1.0);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("HNSW-based Dimensionality Reduction Demo", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;
    
    chart.configure_mesh().draw()?;
    
    // Color points by their original cluster
    for (idx, point) in embeddings.iter().enumerate() {
        let color = if idx < n_points_per_cluster {
            // First cluster - Red
            RGBColor(200, 50, 50)
        } else if idx < 2 * n_points_per_cluster {
            // Second cluster - Green
            RGBColor(50, 200, 50)
        } else if idx < 3 * n_points_per_cluster {
            // Third cluster - Blue
            RGBColor(50, 50, 200)
        } else {
            // Noise points - Gray
            RGBColor(150, 150, 150)
        };
        
        chart.draw_series(PointSeries::of_element(
            vec![(point[0], point[1])],
            5,
            color.mix(0.7).filled(),
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st),
        ))?;
    }
    
    // Add a legend
    chart.configure_series_labels()
        .background_style(WHITE.filled())
        .border_style(BLACK)
        .draw()?;
    
    root.present()?;
    println!("Output saved as hnsw_embedding_demo.png");
    
    Ok(())
} 