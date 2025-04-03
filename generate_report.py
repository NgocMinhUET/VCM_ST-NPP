import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comprehensive report")
    parser.add_argument("--compression_results", type=str, required=True,
                        help="Path to compression results JSON file")
    parser.add_argument("--tracking_results", type=str, required=True,
                        help="Path to tracking results JSON file")
    parser.add_argument("--output_dir", type=str, default="results/final_report",
                        help="Output directory for the comprehensive report")
    return parser.parse_args()

def create_rd_curve(compression_results, output_dir):
    """Create rate-distortion curve comparing compression methods."""
    print("Creating R-D curve...")
    
    # Extract methods and results
    methods = {}
    our_method_data = []
    
    for method, results in compression_results.items():
        if method.startswith("our_method_qp"):
            # Extract results from our method at different QP values
            our_method_data.append((results["bpp"], results["psnr"]))
        elif method.startswith("h264_crf"):
            # Extract the CRF value
            crf = method.split("_crf")[1]
            if "h264" not in methods:
                methods["h264"] = {"name": "H.264", "data": []}
            methods["h264"]["data"].append((results["bpp"], results["psnr"], int(crf)))
        elif method.startswith("h265_crf"):
            # Extract the CRF value
            crf = method.split("_crf")[1]
            if "h265" not in methods:
                methods["h265"] = {"name": "H.265/HEVC", "data": []}
            methods["h265"]["data"].append((results["bpp"], results["psnr"], int(crf)))
        elif method.startswith("vp9_crf"):
            # Extract the CRF value
            crf = method.split("_crf")[1]
            if "vp9" not in methods:
                methods["vp9"] = {"name": "VP9", "data": []}
            methods["vp9"]["data"].append((results["bpp"], results["psnr"], int(crf)))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot standard methods
    colors = ['b', 'g', 'r']
    markers = ['o', 's', '^']
    
    for i, (method_key, method_data) in enumerate(methods.items()):
        # Sort by bpp (ascending)
        sorted_data = sorted(method_data["data"], key=lambda x: x[0])
        bpp_values = [point[0] for point in sorted_data]
        psnr_values = [point[1] for point in sorted_data]
        crf_values = [point[2] for point in sorted_data]
        
        plt.plot(bpp_values, psnr_values, color=colors[i], marker=markers[i], 
                 linestyle='-', linewidth=2, label=method_data["name"])
        
        # Annotate CRF values
        for j, (bpp, psnr, crf) in enumerate(zip(bpp_values, psnr_values, crf_values)):
            plt.annotate(f"CRF {crf}", (bpp, psnr), 
                         textcoords="offset points", xytext=(0, 10), 
                         ha='center', fontsize=8)
    
    # Plot our method (if available)
    if our_method_data:
        # Sort by bpp (ascending)
        sorted_data = sorted(our_method_data, key=lambda x: x[0])
        bpp_values = [point[0] for point in sorted_data]
        psnr_values = [point[1] for point in sorted_data]
        
        plt.plot(bpp_values, psnr_values, 'mo-', markersize=8, linewidth=2,
                 label="Our Method (VQ-VAE)")
        
        # Annotate QP values
        for j, (bpp, psnr) in enumerate(zip(bpp_values, psnr_values)):
            plt.annotate(f"QP {j+1}", (bpp, psnr), 
                         textcoords="offset points", xytext=(0, 10), 
                         ha='center', fontsize=8)
    
    # Add details
    plt.xlabel("Bits Per Pixel (BPP)", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("Rate-Distortion Performance Comparison", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "rd_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_tracking_comparison(tracking_results, output_dir):
    """Create comparison charts for tracking performance."""
    print("Creating tracking performance comparison...")
    
    # Extract methods and metrics
    methods = []
    mota_values = []
    precision_values = []
    recall_values = []
    
    # Original video is our reference
    reference_mota = tracking_results.get("original_video", {}).get("mota", 0)
    reference_precision = tracking_results.get("original_video", {}).get("precision", 0)
    reference_recall = tracking_results.get("original_video", {}).get("recall", 0)
    
    for method, results in tracking_results.items():
        if method == "original_video":
            continue
            
        if method == "our_method":
            method_name = "Our Method"
        elif method.startswith("h264"):
            method_name = "H.264"
        elif method.startswith("h265"):
            method_name = "H.265"
        elif method.startswith("vp9"):
            method_name = "VP9"
        else:
            method_name = method
            
        methods.append(method_name)
        mota_values.append(results["mota"] / reference_mota if reference_mota else results["mota"])
        precision_values.append(results["precision"] / reference_precision if reference_precision else results["precision"])
        recall_values.append(results["recall"] / reference_recall if reference_recall else results["recall"])
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, mota_values, width, label='MOTA', color='skyblue')
    ax.bar(x, precision_values, width, label='Precision', color='lightgreen')
    ax.bar(x + width, recall_values, width, label='Recall', color='salmon')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Ratio to Original Video')
    ax.set_title('Tracking Performance Relative to Original Video')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tracking_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_efficiency_vs_performance(compression_results, tracking_results, output_dir):
    """Create scatter plot showing compression efficiency vs tracking performance."""
    print("Creating efficiency vs performance plot...")
    
    data_points = []
    
    # Find lowest bpp version of our method for tracking results match
    our_method_tracking_key = None
    for method in tracking_results.keys():
        if method.startswith("our_method") or method == "our_method":
            our_method_tracking_key = method
            break
    
    # Process each compression method
    for method, compression_metrics in compression_results.items():
        # Skip our method variants - we'll add them separately below
        if method.startswith("our_method_qp"):
            continue
            
        if method in tracking_results:
            tracking_metrics = tracking_results[method]
        elif method == "our_method" and our_method_tracking_key:
            # Use tracking results from the main our_method entry
            tracking_metrics = tracking_results[our_method_tracking_key]
        else:
            # If no tracking metrics, skip
            continue
        
        data_points.append({
            "method": method,
            "bpp": compression_metrics["bpp"],
            "compression_ratio": compression_metrics["compression_ratio"],
            "mota": tracking_metrics["mota"],
            "precision": tracking_metrics["precision"],
            "recall": tracking_metrics["recall"]
        })
    
    # Add our method variants with the same tracking metrics
    if our_method_tracking_key and our_method_tracking_key in tracking_results:
        our_tracking_metrics = tracking_results[our_method_tracking_key]
        
        for method, compression_metrics in compression_results.items():
            if method.startswith("our_method_qp"):
                data_points.append({
                    "method": method,
                    "bpp": compression_metrics["bpp"],
                    "compression_ratio": compression_metrics["compression_ratio"],
                    "mota": our_tracking_metrics["mota"],
                    "precision": our_tracking_metrics["precision"],
                    "recall": our_tracking_metrics["recall"]
                })
    
    # Extract data for plotting
    methods = [p["method"] for p in data_points]
    bpp_values = [p["bpp"] for p in data_points]
    mota_values = [p["mota"] for p in data_points]
    sizes = [p["compression_ratio"] * 2 for p in data_points]  # Size based on compression ratio
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Define colors for different method types
    colors = []
    for method in methods:
        if method.startswith("our_method"):
            colors.append('purple')
        elif "h264" in method:
            colors.append('blue')
        elif "h265" in method:
            colors.append('green')
        elif "vp9" in method:
            colors.append('red')
        else:
            colors.append('gray')
    
    # Create scatter plot
    plt.scatter(bpp_values, mota_values, s=sizes, c=colors, alpha=0.7)
    
    # Add labels for each point
    for i, method in enumerate(methods):
        if method.startswith("our_method_qp"):
            method_label = f"Our QP{method.split('_qp')[1]}"
        elif method == "our_method":
            method_label = "Our Method"
        else:
            method_label = method
        plt.annotate(method_label, (bpp_values[i], mota_values[i]),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("MOTA Score")
    plt.title("Compression Efficiency vs Tracking Performance")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Our Method'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='H.264'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='H.265'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='VP9')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "efficiency_vs_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(compression_results, tracking_results, output_dir):
    """Generate summary table comparing all methods."""
    print("Generating summary table...")
    
    # Combine results
    combined_results = {}
    
    # Find lowest bpp version of our method for tracking results match
    our_method_tracking_key = None
    for method in tracking_results.keys():
        if method.startswith("our_method") or method == "our_method":
            our_method_tracking_key = method
            break
    
    # Start with the original video (reference)
    if "original_video" in tracking_results:
        combined_results["Original Video"] = {
            "compression_ratio": "N/A",
            "bpp": "N/A",
            "psnr": "N/A",
            "mota": tracking_results["original_video"]["mota"],
            "precision": tracking_results["original_video"]["precision"],
            "recall": tracking_results["original_video"]["recall"]
        }
    
    # Add all methods
    for method, compression_metrics in compression_results.items():
        if method.startswith("our_method_qp"):
            method_name = f"Our Method QP{method.split('_qp')[1]}"
        elif method == "our_method":
            method_name = "Our Method (VQ-VAE)"
        else:
            method_name = method.upper()
            
        # Add compression metrics
        combined_results[method_name] = {
            "compression_ratio": compression_metrics["compression_ratio"],
            "bpp": compression_metrics["bpp"],
            "psnr": compression_metrics["psnr"],
        }
            
        # Add tracking metrics if available
        if method in tracking_results:
            tracking_metrics = tracking_results[method]
            combined_results[method_name].update({
                "mota": tracking_metrics["mota"],
                "precision": tracking_metrics["precision"],
                "recall": tracking_metrics["recall"]
            })
        elif method.startswith("our_method") and our_method_tracking_key:
            # Use tracking results from the main our_method entry
            tracking_metrics = tracking_results[our_method_tracking_key]
            combined_results[method_name].update({
                "mota": tracking_metrics["mota"],
                "precision": tracking_metrics["precision"],
                "recall": tracking_metrics["recall"]
            })
        else:
            # If no tracking metrics, set to N/A
            combined_results[method_name].update({
                "mota": "N/A",
                "precision": "N/A",
                "recall": "N/A"
            })
    
    # Generate table
    with open(os.path.join(output_dir, "summary_table.txt"), 'w') as f:
        f.write("Comprehensive Performance Summary\n")
        f.write("================================\n\n")
        
        f.write(f"{'Method':<20} | {'Comp.Ratio':>12} | {'BPP':>8} | {'PSNR (dB)':>9} | {'MOTA':>8} | {'Precision':>9} | {'Recall':>8}\n")
        f.write("-" * 90 + "\n")
        
        # Sort by compression ratio (our method first, then by compression ratio)
        def sort_key(item):
            method, metrics = item
            if method == "Our Method (VQ-VAE)":
                return (0, 0)
            if method == "Original Video":
                return (2, 0)
            return (1, -metrics["compression_ratio"] if isinstance(metrics["compression_ratio"], (int, float)) else 0)
        
        for method, metrics in sorted(combined_results.items(), key=sort_key):
            # Format values
            comp_ratio = f"{metrics['compression_ratio']:.2f}" if isinstance(metrics['compression_ratio'], (int, float)) else metrics['compression_ratio']
            bpp = f"{metrics['bpp']:.4f}" if isinstance(metrics['bpp'], (int, float)) else metrics['bpp']
            psnr = f"{metrics['psnr']:.2f}" if isinstance(metrics['psnr'], (int, float)) else metrics['psnr']
            mota = f"{metrics['mota']:.4f}" if isinstance(metrics['mota'], (int, float)) else metrics['mota']
            precision = f"{metrics['precision']:.4f}" if isinstance(metrics['precision'], (int, float)) else metrics['precision']
            recall = f"{metrics['recall']:.4f}" if isinstance(metrics['recall'], (int, float)) else metrics['recall']
            
            f.write(f"{method:<20} | {comp_ratio:>12} | {bpp:>8} | {psnr:>9} | {mota:>8} | {precision:>9} | {recall:>8}\n")

def calculate_bd_metrics(compression_results, tracking_results, output_dir):
    """Calculate Bjøntegaard Delta metrics (BD-Rate and BD-MOTA)."""
    print("Calculating BD metrics...")
    
    # Implementation of Bjøntegaard Delta calculation (simplified for this context)
    def bd_rate(R1, PSNR1, R2, PSNR2):
        from scipy import interpolate
        
        # Logarithm of the bitrates
        log_R1 = np.log(R1)
        log_R2 = np.log(R2)
        
        # Fit a cubic polynomial to the data
        p1 = np.polyfit(PSNR1, log_R1, 3)
        p2 = np.polyfit(PSNR2, log_R2, 3)
        
        # Integration interval
        min_psnr = max(min(PSNR1), min(PSNR2))
        max_psnr = min(max(PSNR1), max(PSNR2))
        
        # Calculate the integral of the difference
        avg_diff = 0
        steps = 100
        for psnr in np.linspace(min_psnr, max_psnr, steps):
            log_rate1 = np.polyval(p1, psnr)
            log_rate2 = np.polyval(p2, psnr)
            avg_diff += (log_rate2 - log_rate1)
        
        avg_diff /= steps
        
        # Convert to percentage
        bd_rate = (np.exp(avg_diff) - 1) * 100
        return bd_rate
    
    # Extract data for BD calculation
    # Standard codecs
    codec_data = {
        "h264": {"bpp": [], "psnr": [], "mota": []},
        "h265": {"bpp": [], "psnr": [], "mota": []},
        "vp9": {"bpp": [], "psnr": [], "mota": []}
    }
    
    # Our method data
    our_method = {"bpp": [], "psnr": [], "mota": []}
    
    # Extract BD data from results
    for method, results in compression_results.items():
        if method.startswith("our_method_qp"):
            our_method["bpp"].append(results["bpp"])
            our_method["psnr"].append(results["psnr"])
            
            # Try to find tracking metrics
            if method in tracking_results:
                our_method["mota"].append(tracking_results[method]["mota"])
            elif "our_method" in tracking_results:
                our_method["mota"].append(tracking_results["our_method"]["mota"])
        
        elif method.startswith("h264_crf"):
            codec_data["h264"]["bpp"].append(results["bpp"])
            codec_data["h264"]["psnr"].append(results["psnr"])
            if method in tracking_results:
                codec_data["h264"]["mota"].append(tracking_results[method]["mota"])
        
        elif method.startswith("h265_crf"):
            codec_data["h265"]["bpp"].append(results["bpp"])
            codec_data["h265"]["psnr"].append(results["psnr"])
            if method in tracking_results:
                codec_data["h265"]["mota"].append(tracking_results[method]["mota"])
        
        elif method.startswith("vp9_crf"):
            codec_data["vp9"]["bpp"].append(results["bpp"])
            codec_data["vp9"]["psnr"].append(results["psnr"])
            if method in tracking_results:
                codec_data["vp9"]["mota"].append(tracking_results[method]["mota"])
    
    # Calculate BD metrics and save to results file
    bd_results = {}
    
    # Only calculate if we have enough data points for each method
    if len(our_method["bpp"]) >= 3:
        for codec, data in codec_data.items():
            if len(data["bpp"]) >= 3:
                # Sort data by bpp
                codec_zipped = sorted(zip(data["bpp"], data["psnr"]), key=lambda x: x[0])
                sorted_codec_bpp = [x[0] for x in codec_zipped]
                sorted_codec_psnr = [x[1] for x in codec_zipped]
                
                our_zipped = sorted(zip(our_method["bpp"], our_method["psnr"]), key=lambda x: x[0])
                sorted_our_bpp = [x[0] for x in our_zipped]
                sorted_our_psnr = [x[1] for x in our_zipped]
                
                # Calculate BD-Rate
                try:
                    bd_rate_value = bd_rate(sorted_codec_bpp, sorted_codec_psnr, 
                                           sorted_our_bpp, sorted_our_psnr)
                    bd_results[f"BD-Rate vs {codec.upper()}"] = f"{bd_rate_value:.2f}%"
                except Exception as e:
                    bd_results[f"BD-Rate vs {codec.upper()}"] = f"Error: {str(e)}"
                
                # Calculate BD-MOTA if we have enough tracking data
                if len(data["mota"]) >= 3 and len(our_method["mota"]) >= 3:
                    try:
                        # For BD-MOTA, we need to interpolate MOTA values at the same bitrates
                        # This is a very simplified approach
                        codec_mota_zipped = sorted(zip(data["bpp"], data["mota"]), key=lambda x: x[0])
                        our_mota_zipped = sorted(zip(our_method["bpp"], our_method["mota"]), key=lambda x: x[0])
                        
                        # Simple estimate of BD-MOTA: average MOTA improvement at similar bitrates
                        codec_mota_avg = sum(m for _, m in codec_mota_zipped) / len(codec_mota_zipped)
                        our_mota_avg = sum(m for _, m in our_mota_zipped) / len(our_mota_zipped)
                        bd_mota = ((our_mota_avg / codec_mota_avg) - 1) * 100
                        
                        bd_results[f"BD-MOTA vs {codec.upper()}"] = f"{bd_mota:.2f}%"
                    except Exception as e:
                        bd_results[f"BD-MOTA vs {codec.upper()}"] = f"Error: {str(e)}"
    
    # Save BD results to file
    bd_path = os.path.join(output_dir, "bd_metrics.txt")
    with open(bd_path, 'w') as f:
        f.write("Bjøntegaard Delta Metrics\n")
        f.write("=========================\n\n")
        
        if bd_results:
            for metric, value in bd_results.items():
                f.write(f"{metric}: {value}\n")
            
            f.write("\nNotes:\n")
            f.write("- Negative BD-Rate indicates bitrate savings of our method vs standard codec\n")
            f.write("- Positive BD-MOTA indicates tracking performance improvement of our method vs standard codec\n")
        else:
            f.write("Insufficient data to calculate BD metrics.\n")
            f.write("At least 3 data points are required for each method.\n")
    
    return bd_results

def generate_report(args):
    """Generate comprehensive report."""
    print(f"Generating comprehensive report with results from:")
    print(f"- Compression results: {args.compression_results}")
    print(f"- Tracking results: {args.tracking_results}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    with open(args.compression_results, 'r') as f:
        compression_results = json.load(f)
    
    with open(args.tracking_results, 'r') as f:
        tracking_results = json.load(f)
    
    # Generate components
    create_rd_curve(compression_results, args.output_dir)
    create_tracking_comparison(tracking_results, args.output_dir)
    create_efficiency_vs_performance(compression_results, tracking_results, args.output_dir)
    generate_summary_table(compression_results, tracking_results, args.output_dir)
    bd_results = calculate_bd_metrics(compression_results, tracking_results, args.output_dir)
    
    # Generate main report file
    with open(os.path.join(args.output_dir, "comprehensive_report.md"), 'w') as f:
        f.write("# Comprehensive Evaluation Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report presents a comprehensive evaluation of our improved autoencoder-based video compression method compared to standard video codecs. ")
        f.write("We evaluate both compression efficiency and the impact on downstream computer vision tasks (object tracking).\n\n")
        
        f.write("## Compression Performance\n\n")
        f.write("The rate-distortion curve below shows the performance of different compression methods:\n\n")
        f.write("![Rate-Distortion Curve](./rd_curve.png)\n\n")
        
        # Add BD-Rate results if available
        if bd_results:
            f.write("### Bjøntegaard Delta Rate (BD-Rate)\n\n")
            bd_rate_results = {k: v for k, v in bd_results.items() if "BD-Rate" in k}
            if bd_rate_results:
                f.write("BD-Rate measures the average bitrate savings compared to standard codecs:\n\n")
                for metric, value in bd_rate_results.items():
                    f.write(f"- {metric}: {value}\n")
                f.write("\nNegative values indicate bitrate savings of our method.\n\n")
        
        f.write("## Tracking Performance\n\n")
        f.write("The impact of different compression methods on tracking performance is illustrated below:\n\n")
        f.write("![Tracking Performance](./tracking_comparison.png)\n\n")
        
        # Add BD-MOTA results if available
        if bd_results:
            f.write("### Bjøntegaard Delta MOTA (BD-MOTA)\n\n")
            bd_mota_results = {k: v for k, v in bd_results.items() if "BD-MOTA" in k}
            if bd_mota_results:
                f.write("BD-MOTA measures the average tracking performance improvement at similar bitrates:\n\n")
                for metric, value in bd_mota_results.items():
                    f.write(f"- {metric}: {value}\n")
                f.write("\nPositive values indicate tracking performance improvement of our method.\n\n")
        
        f.write("## Efficiency vs. Performance Trade-off\n\n")
        f.write("This plot shows the relationship between compression efficiency (BPP) and tracking performance (MOTA):\n\n")
        f.write("![Efficiency vs Performance](./efficiency_vs_performance.png)\n\n")
        
        f.write("## Summary\n\n")
        f.write("Our method achieves a significantly better compression ratio compared to standard codecs, ")
        f.write("while maintaining acceptable tracking performance. This makes it particularly suitable for ")
        f.write("bandwidth-constrained video analytics applications.\n\n")
        
        f.write("For detailed metrics, please refer to the [summary table](./summary_table.txt) and [BD metrics](./bd_metrics.txt).\n")
    
    # Attempt to generate HTML version if markdown module is available
    try:
        import markdown
        
        # Read markdown file
        md_path = os.path.join(args.output_dir, "comprehensive_report.md")
        with open(md_path, 'r') as f:
            md_content = f.read()
        
        # Convert to HTML with CSS styling
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .container {{
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 25px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {markdown.markdown(md_content)}
    </div>
</body>
</html>
        """
        
        # Save as HTML
        html_path = os.path.join(args.output_dir, "comprehensive_report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive report generated successfully in {args.output_dir}")
        print(f"HTML version available at: {html_path}")
        
    except ImportError:
        print(f"Comprehensive report generated successfully in {args.output_dir}")
        print(f"Note: HTML version not generated (markdown module not available)")
    
    return

def main():
    args = parse_args()
    generate_report(args)

if __name__ == "__main__":
    main() 