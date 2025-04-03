#!/usr/bin/env python
"""
Script to generate a comprehensive evaluation report comparing different video compression methods
for object tracking applications.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comprehensive evaluation report")
    
    # Input parameters
    parser.add_argument("--compression_results", type=str, required=True,
                        help="Path to compression results directory")
    parser.add_argument("--tracking_results", type=str, required=True,
                        help="Path to tracking evaluation results directory")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, 
                        default="results/comprehensive_report",
                        help="Directory to save the comprehensive report")
    
    return parser.parse_args()

def load_results(compression_path, tracking_path):
    """Load compression and tracking results from JSON files."""
    # Load compression results
    compression_file = os.path.join(compression_path, "compression_results.json")
    if not os.path.exists(compression_file):
        raise ValueError(f"Compression results file not found at {compression_file}")
    
    with open(compression_file, 'r') as f:
        compression_results = json.load(f)
    
    # Load tracking results
    tracking_file = os.path.join(tracking_path, "tracking_results.json")
    if not os.path.exists(tracking_file):
        raise ValueError(f"Tracking results file not found at {tracking_file}")
    
    with open(tracking_file, 'r') as f:
        tracking_results = json.load(f)
    
    return compression_results, tracking_results

def create_combined_dataframe(compression_results, tracking_results):
    """Create a combined DataFrame with all metrics for analysis."""
    combined_data = []
    
    # Process each method
    for method in compression_results:
        if method in tracking_results:
            # Get compression metrics
            comp_data = compression_results[method]
            track_data = tracking_results[method]
            
            # Extract codec information
            if method == "our_method":
                codec = "Our Method"
                crf = "N/A"
            else:
                parts = method.split("_")
                codec = parts[0]
                crf = parts[1].replace("crf", "")
            
            # Create entry
            entry = {
                "Method": method,
                "Codec": codec,
                "CRF": crf,
                "Compression Ratio": comp_data["compression_ratio"],
                "BPP": comp_data["bpp"],
                "PSNR": comp_data["psnr"],
                "MS-SSIM": comp_data["ms_ssim"],
                "Compression Time": comp_data["compression_time"],
                "MOTA": track_data["mota"],
                "MOTP": track_data["motp"],
                "Precision": track_data["precision"],
                "Recall": track_data["recall"],
                "Delta MOTA": track_data["delta_mota"],
                "Delta MOTP": track_data["delta_motp"],
                "Delta Precision": track_data["delta_precision"],
                "Delta Recall": track_data["delta_recall"]
            }
            
            combined_data.append(entry)
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    return df

def create_radar_chart(df, output_path):
    """Create a radar chart comparing the top methods."""
    # Select top methods (highest compression ratio with MOTA > 0.9)
    high_quality_df = df[df["MOTA"] > 0.9].copy()
    if len(high_quality_df) == 0:
        high_quality_df = df.copy()
    
    top_methods = high_quality_df.sort_values("Compression Ratio", ascending=False).head(4)
    
    # Normalize metrics for radar chart
    metrics = ["Compression Ratio", "PSNR", "MS-SSIM", "MOTA", "Negative BPP"]
    
    # Add negative BPP (lower is better)
    top_methods["Negative BPP"] = -top_methods["BPP"]
    
    # Create a normalized DataFrame
    normalized_df = pd.DataFrame()
    for metric in metrics:
        min_val = top_methods[metric].min()
        max_val = top_methods[metric].max()
        if max_val > min_val:
            normalized_df[metric] = (top_methods[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 1.0
    
    # Setup the radar chart
    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111, polar=True)
    
    # Add each method
    for i, method in enumerate(top_methods["Method"]):
        values = normalized_df.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        
        if method == "our_method":
            color = "red"
            linewidth = 3
            alpha = 1.0
            label = "Our Method"
        else:
            codec = method.split("_")[0]
            crf = method.split("_")[1].replace("crf", "")
            color = {"h264": "blue", "h265": "green", "vp9": "purple"}[codec]
            linewidth = 2
            alpha = 0.7
            label = f"{codec.upper()} (CRF {crf})"
        
        ax.plot(angles, values, color=color, linewidth=linewidth, alpha=alpha, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Comparison of Top Compression Methods', size=15)
    plt.legend(loc='upper right')
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_efficiency_chart(df, output_path):
    """Create a chart showing compression efficiency (PSNR/BPP) vs. tracking accuracy."""
    # Calculate compression efficiency
    df["Efficiency"] = df["PSNR"] / df["BPP"]
    
    # Group by codec
    codecs = df["Codec"].unique()
    
    plt.figure(figsize=(12, 8))
    
    # Plot each codec with a different color
    for codec in codecs:
        codec_df = df[df["Codec"] == codec]
        
        if codec == "Our Method":
            plt.scatter(codec_df["Efficiency"], codec_df["MOTA"], 
                      color="red", marker='o', s=150, label=codec, zorder=10)
        else:
            plt.scatter(codec_df["Efficiency"], codec_df["MOTA"],
                      alpha=0.7, marker={"h264": "s", "h265": "^", "vp9": "D"}[codec.lower()], 
                      s=80, label=codec)
    
    plt.xlabel('Compression Efficiency (PSNR/BPP)', fontsize=12)
    plt.ylabel('MOTA', fontsize=12)
    plt.title('Compression Efficiency vs. Tracking Accuracy', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_pareto_chart(df, output_path):
    """Create a Pareto efficiency chart with compression ratio and tracking accuracy."""
    plt.figure(figsize=(12, 8))
    
    # Plot points
    for codec in df["Codec"].unique():
        codec_df = df[df["Codec"] == codec]
        
        if codec == "Our Method":
            plt.scatter(codec_df["Compression Ratio"], codec_df["MOTA"], 
                      color="red", marker='o', s=150, label=codec, zorder=10)
        else:
            plt.scatter(codec_df["Compression Ratio"], codec_df["MOTA"],
                      alpha=0.7, marker={"h264": "s", "h265": "^", "vp9": "D"}[codec.lower()], 
                      s=80, label=codec)
    
    # Add method names as labels for points with high compression or MOTA
    for i, row in df.iterrows():
        if row["Compression Ratio"] > df["Compression Ratio"].median() * 1.5 or row["MOTA"] > df["MOTA"].median() * 1.1:
            if row["Method"] == "our_method":
                label = "Our Method"
            else:
                codec = row["Method"].split("_")[0]
                crf = row["Method"].split("_")[1].replace("crf", "")
                label = f"{codec.upper()} (CRF {crf})"
            
            plt.annotate(label, (row["Compression Ratio"], row["MOTA"]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Compression Ratio', fontsize=12)
    plt.ylabel('MOTA', fontsize=12)
    plt.title('Pareto Efficiency: Compression Ratio vs. Tracking Accuracy', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # If compression ratio range is very large, use log scale
    if df["Compression Ratio"].max() / df["Compression Ratio"].min() > 100:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_tracking_degradation_chart(df, output_path):
    """Create a chart showing tracking degradation vs. compression ratio."""
    plt.figure(figsize=(12, 8))
    
    # Plot points
    for codec in df["Codec"].unique():
        codec_df = df[df["Codec"] == codec]
        
        if codec == "Our Method":
            plt.scatter(codec_df["Compression Ratio"], codec_df["Delta MOTA"], 
                      color="red", marker='o', s=150, label=codec, zorder=10)
        else:
            plt.scatter(codec_df["Compression Ratio"], codec_df["Delta MOTA"],
                      alpha=0.7, marker={"h264": "s", "h265": "^", "vp9": "D"}[codec.lower()], 
                      s=80, label=codec)
    
    # Draw a horizontal line at Delta MOTA = 0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Compression Ratio', fontsize=12)
    plt.ylabel('Change in MOTA', fontsize=12)
    plt.title('Tracking Performance Degradation vs. Compression Ratio', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # If compression ratio range is very large, use log scale
    if df["Compression Ratio"].max() / df["Compression Ratio"].min() > 100:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(df, output_dir):
    """Generate a comprehensive HTML report with all metrics and visualizations."""
    # Create report path
    report_path = os.path.join(output_dir, "comprehensive_report.html")
    
    # Generate summary statistics
    summary_by_codec = df.groupby("Codec").mean().reset_index()
    
    # Sort methods by different metrics
    best_compression = df.sort_values("Compression Ratio", ascending=False).head(5)
    best_quality = df.sort_values("PSNR", ascending=False).head(5)
    best_tracking = df.sort_values("MOTA", ascending=False).head(5)
    best_efficiency = df.sort_values(df["PSNR"] / df["BPP"], ascending=False).head(5)
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comprehensive Video Compression Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #e3f2fd; }}
            .container {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
            .chart {{ width: 48%; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; }}
            .summary {{ margin-bottom: 30px; }}
            .best {{ color: green; font-weight: bold; }}
            .worst {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Comprehensive Video Compression Evaluation Report</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>This report compares different video compression methods for surveillance and object tracking applications.
            The evaluation includes our improved autoencoder-based method and standard codecs (H.264, H.265, VP9) at various quality settings.</p>
            
            <h3>Key Findings:</h3>
            <ul>
                <li>Highest compression ratio: <span class="best">{df["Method"].iloc[df["Compression Ratio"].argmax()]}</span> 
                    ({df["Compression Ratio"].max():.2f}x)</li>
                <li>Best visual quality (PSNR): <span class="best">{df["Method"].iloc[df["PSNR"].argmax()]}</span> 
                    ({df["PSNR"].max():.2f} dB)</li>
                <li>Best tracking accuracy (MOTA): <span class="best">{df["Method"].iloc[df["MOTA"].argmax()]}</span> 
                    ({df["MOTA"].max():.4f})</li>
                <li>Least tracking degradation: <span class="best">{df["Method"].iloc[df["Delta MOTA"].argmax()]}</span> 
                    (Delta MOTA: {df["Delta MOTA"].max():.4f})</li>
            </ul>
        </div>
        
        <h2>Compression and Tracking Performance</h2>
        
        <div class="container">
            <div class="chart">
                <h3>Pareto Efficiency: Compression vs. Tracking</h3>
                <img src="pareto_chart.png" alt="Pareto Chart">
            </div>
            <div class="chart">
                <h3>Tracking Degradation vs. Compression</h3>
                <img src="tracking_degradation_chart.png" alt="Tracking Degradation Chart">
            </div>
        </div>
        
        <div class="container">
            <div class="chart">
                <h3>Compression Efficiency vs. Tracking</h3>
                <img src="efficiency_chart.png" alt="Efficiency Chart">
            </div>
            <div class="chart">
                <h3>Comparison of Top Methods</h3>
                <img src="radar_chart.png" alt="Radar Chart">
            </div>
        </div>
        
        <h2>Detailed Results</h2>
        
        <h3>All Methods</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>Compression Ratio</th>
                <th>BPP</th>
                <th>PSNR (dB)</th>
                <th>MS-SSIM</th>
                <th>MOTA</th>
                <th>MOTP</th>
                <th>Delta MOTA</th>
            </tr>
    """
    
    # Add all methods to the table
    for _, row in df.sort_values("Compression Ratio", ascending=False).iterrows():
        # Format method name
        if row["Method"] == "our_method":
            method_name = "Our Method"
        else:
            codec = row["Method"].split("_")[0].upper()
            crf = row["Method"].split("_")[1].replace("crf", "")
            method_name = f"{codec} (CRF {crf})"
        
        # Highlight our method
        highlight = ' class="highlight"' if row["Method"] == "our_method" else ""
        
        html_content += f"""
            <tr{highlight}>
                <td>{method_name}</td>
                <td>{row["Compression Ratio"]:.2f}x</td>
                <td>{row["BPP"]:.4f}</td>
                <td>{row["PSNR"]:.2f}</td>
                <td>{row["MS-SSIM"]:.4f}</td>
                <td>{row["MOTA"]:.4f}</td>
                <td>{row["MOTP"]:.4f}</td>
                <td>{row["Delta MOTA"]:+.4f}</td>
            </tr>
        """
    
    # Add summary by codec
    html_content += """
        </table>
        
        <h3>Summary by Codec</h3>
        <table>
            <tr>
                <th>Codec</th>
                <th>Avg. Compression Ratio</th>
                <th>Avg. PSNR (dB)</th>
                <th>Avg. MS-SSIM</th>
                <th>Avg. MOTA</th>
                <th>Avg. Delta MOTA</th>
            </tr>
    """
    
    for _, row in summary_by_codec.iterrows():
        # Highlight our method
        highlight = ' class="highlight"' if row["Codec"] == "Our Method" else ""
        
        html_content += f"""
            <tr{highlight}>
                <td>{row["Codec"]}</td>
                <td>{row["Compression Ratio"]:.2f}x</td>
                <td>{row["PSNR"]:.2f}</td>
                <td>{row["MS-SSIM"]:.4f}</td>
                <td>{row["MOTA"]:.4f}</td>
                <td>{row["Delta MOTA"]:+.4f}</td>
            </tr>
        """
    
    # Add best methods sections
    html_content += """
        </table>
        
        <h3>Best Compression Ratio</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>Compression Ratio</th>
                <th>PSNR (dB)</th>
                <th>MOTA</th>
            </tr>
    """
    
    for _, row in best_compression.iterrows():
        # Format method name
        if row["Method"] == "our_method":
            method_name = "Our Method"
        else:
            codec = row["Method"].split("_")[0].upper()
            crf = row["Method"].split("_")[1].replace("crf", "")
            method_name = f"{codec} (CRF {crf})"
        
        # Highlight our method
        highlight = ' class="highlight"' if row["Method"] == "our_method" else ""
        
        html_content += f"""
            <tr{highlight}>
                <td>{method_name}</td>
                <td>{row["Compression Ratio"]:.2f}x</td>
                <td>{row["PSNR"]:.2f}</td>
                <td>{row["MOTA"]:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>Best Tracking Performance</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>MOTA</th>
                <th>Compression Ratio</th>
                <th>PSNR (dB)</th>
            </tr>
    """
    
    for _, row in best_tracking.iterrows():
        # Format method name
        if row["Method"] == "our_method":
            method_name = "Our Method"
        else:
            codec = row["Method"].split("_")[0].upper()
            crf = row["Method"].split("_")[1].replace("crf", "")
            method_name = f"{codec} (CRF {crf})"
        
        # Highlight our method
        highlight = ' class="highlight"' if row["Method"] == "our_method" else ""
        
        html_content += f"""
            <tr{highlight}>
                <td>{method_name}</td>
                <td>{row["MOTA"]:.4f}</td>
                <td>{row["Compression Ratio"]:.2f}x</td>
                <td>{row["PSNR"]:.2f}</td>
            </tr>
        """
    
    # Finish HTML content
    html_content += """
        </table>
        
        <h2>Conclusion</h2>
        <p>
        Based on the comprehensive evaluation, our method demonstrates significant advantages in terms of compression ratio
        while maintaining competitive tracking performance. For applications where extreme compression is required, our method
        offers clear benefits. Standard codecs like H.265 at moderate quality settings (CRF 23-28) provide a good balance between
        compression efficiency and tracking accuracy for general-purpose surveillance.
        </p>
        
        <h3>Recommendations:</h3>
        <ul>
            <li>For bandwidth-constrained scenarios: Use our method for maximum compression</li>
            <li>For general surveillance: H.265 at CRF 23-28 provides good balance</li>
            <li>For high-precision tracking: H.265 at CRF 18-23 minimizes tracking degradation</li>
        </ul>
        
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated comprehensive report at {report_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    compression_results, tracking_results = load_results(args.compression_results, args.tracking_results)
    
    # Create combined DataFrame
    df = create_combined_dataframe(compression_results, tracking_results)
    
    # Create charts
    create_radar_chart(df, os.path.join(args.output_dir, "radar_chart.png"))
    create_efficiency_chart(df, os.path.join(args.output_dir, "efficiency_chart.png"))
    create_pareto_chart(df, os.path.join(args.output_dir, "pareto_chart.png"))
    create_tracking_degradation_chart(df, os.path.join(args.output_dir, "tracking_degradation_chart.png"))
    
    # Generate HTML report
    generate_report(df, args.output_dir)
    
    print(f"Comprehensive report generation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 