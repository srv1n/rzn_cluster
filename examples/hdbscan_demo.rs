// Generates synthetic data with 3 clusters and some noise points
// Performs HDBSCAN clustering on this data
// Creates a visualization showing the clusters with different colors
// Prints a report of the clustering results
// The output showed that it found 3 clusters (with IDs 171, 173, and 174) and identified 17 points as outliers, which matches what we would expect from the data generation.

use anyhow::Result;
use plotters::prelude::*;
use rand::distributions::Distribution;
use rand_distr::Normal;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rzn_cluster::clustering::hdbscan_clustering;
use std::collections::HashMap;

fn main() -> Result<()> {
    // Set up a random number generator with a fixed seed for reproducibility
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    
    // Generate 3 clusters of data with different normal distributions
    let n_points_per_cluster = 50;
    let mut data = Vec::new();
    
    // Generate a dataset with 3 clusters in 2D space
    let centers = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)];
    let std_devs = [0.5, 0.8, 0.3];
    
    for i in 0..3 {
        let center_x = centers[i].0;
        let center_y = centers[i].1;
        let std = std_devs[i];
        
        let normal_dist_x = Normal::new(center_x, std).unwrap();
        let normal_dist_y = Normal::new(center_y, std).unwrap();
        
        for _ in 0..n_points_per_cluster {
            let x = normal_dist_x.sample(&mut rng);
            let y = normal_dist_y.sample(&mut rng);
            data.push(vec![x, y]);
        }
    }
    
    // Add some random noise points
    let normal_dist_noise = Normal::new(0.0, 10.0).unwrap();
    for _ in 0..20 {
        let x = normal_dist_noise.sample(&mut rng);
        let y = normal_dist_noise.sample(&mut rng);
        data.push(vec![x, y]);
    }
    
    println!("Generated {} data points", data.len());
    
    // Perform HDBSCAN clustering
    let min_cluster_size = 10;
    let min_samples = 5;
    let result = hdbscan_clustering(&data, min_cluster_size, min_samples, None, None)?;
    
    println!("========= Clustering Report =========");
    println!("Total points: {}", data.len());
    println!("Number of clusters: {}", result.clusters.len());
    println!("Number of outliers: {}", result.outliers.len());
    
    for (cluster_id, points) in result.clusters.iter() {
        println!("Cluster {}: {} points", cluster_id, points.len());
    }
    
    // Plot the clustering results
    let root = BitMapBackend::new("hdbscan_demo.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let x_values: Vec<f64> = data.iter().map(|v| v[0]).collect();
    let y_values: Vec<f64> = data.iter().map(|v| v[1]).collect();
    
    let min_x = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Add some padding to the plot
    let x_range = (min_x - 1.0)..(max_x + 1.0);
    let y_range = (min_y - 1.0)..(max_y + 1.0);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("HDBSCAN Clustering Demo", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;
    
    chart.configure_mesh().draw()?;
    
    // Generate colors for each cluster
    let cluster_count = result.clusters.len();
    let colors: Vec<RGBColor> = (0..cluster_count)
        .map(|i| {
            RGBColor(
                (i * 255 / cluster_count) as u8,
                ((i + cluster_count / 3) * 255 / cluster_count) as u8,
                ((i + 2 * cluster_count / 3) * 255 / cluster_count) as u8,
            )
        })
        .collect();
    
    // Create a mapping of cluster IDs to color indices
    let mut cluster_id_to_color_idx: HashMap<usize, usize> = HashMap::new();
    for (idx, &cluster_id) in result.clusters.keys().enumerate() {
        cluster_id_to_color_idx.insert(cluster_id, idx);
    }
    
    // Plot each cluster with a different color
    for (point_idx, &cluster_id) in result.assignments.iter().enumerate() {
        let color = if cluster_id == 0 {
            BLACK.mix(0.5) // Outliers are black
        } else {
            // Look up the color index for this cluster_id
            let color_idx = cluster_id_to_color_idx.get(&cluster_id).unwrap_or(&0);
            colors[*color_idx].mix(0.7) // Clusters have colors
        };
        
        chart.draw_series(PointSeries::of_element(
            vec![(data[point_idx][0], data[point_idx][1])],
            5,
            color.filled(),
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st),
        ))?;
    }
    
    // Add a legend
    chart.configure_series_labels()
        .background_style(WHITE.filled())
        .border_style(BLACK)
        .draw()?;
    
    root.present()?;
    println!("Output saved as hdbscan_demo.png");
    
    Ok(())
} 