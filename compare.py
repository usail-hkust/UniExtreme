import os
import math
import json
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def get_sorted_method(values, best="max"):

    # 创建(值, 索引)对的列表
    indexed_values = [(value, idx) for idx, value in enumerate(values)]
    
    # 根据排序方式决定排序顺序
    if best == "max":
        # 降序排列
        indexed_values.sort(key=lambda x: (-x[0], x[1]))
    else:
        # 升序排列
        indexed_values.sort(key=lambda x: (x[0], x[1]))
    
    ranking_dict = {}
    current_rank = 1
    same_num = 0
    for i in range(len(indexed_values)):
        if i > 0:
            if indexed_values[i][0] != indexed_values[i-1][0]:
                current_rank += 1 + same_num
                same_num = 0
            else:
                same_num += 1
        method_idx = indexed_values[i][1]
        ranking_dict[method_list[method_idx]] = current_rank
    
    return ranking_dict

def get_compare_and_rank(method2path, CAL_TYPE=True):
    
    universal_data = {}
    type_data = {}
    CAL_TYPE = True
    for method, path in method2path.items():
        with open(path, "r") as f:
            df = json.load(f)
            universal_data[method] = df["universal"]
            type_data[method] = df["type"]
            if len(type_data[method]) == 0:
                CAL_TYPE = False
    
    method_list = list(method2path.keys())
    var_list = list(universal_data["UniExtreme"]["norm"].keys())
    metric_list = list(universal_data["UniExtreme"]["norm"]["all"].keys())
    type_list = list(type_data["UniExtreme"].keys())
    ext_metric_list = list(type_data["UniExtreme"]['Flood']["norm"]["all"].keys())
    compare_dict = {}
    type_compare_dict = {}
    method_avg_rank_universal_by_var = {method: {"norm": {}, "raw": {}} for method in method_list}
    for metric in metric_list:
        for method in method_list:
            method_avg_rank_universal_by_var[method]["norm"][metric] = 0
            method_avg_rank_universal_by_var[method]["raw"][metric] = 0
    method_avg_rank_extreme_by_type = {method: {"norm": {}, "raw": {}} for method in method_list}
    for metric in metric_list:
        for method in method_list:
            method_avg_rank_extreme_by_type[method]["norm"][metric] = 0
            method_avg_rank_extreme_by_type[method]["raw"][metric] = 0
            
    for NORM in ["norm", "raw"]:
        if NORM not in compare_dict:
            compare_dict[NORM] = {}
        for var in var_list:
            if var not in compare_dict[NORM]:
                compare_dict[NORM][var] = {}
            for metric in metric_list:
                local_universal_data = [universal_data[method][NORM][var][metric] for method in method_list]
                compare_dict[NORM][var][metric] = get_sorted_method(local_universal_data, best="max" if "acc_" in metric else "min")
                if var != "all":
                    for method, rank in compare_dict[NORM][var][metric].items():
                        method_avg_rank_universal_by_var[method][NORM][metric] += rank / (len(var_list) - 1)
                        
    if CAL_TYPE:       
        for event_type in type_list: 
            if event_type not in type_compare_dict:
                type_compare_dict[event_type] = {}
            for NORM in ["norm", "raw"]:
                if NORM not in type_compare_dict[event_type]:
                    type_compare_dict[event_type][NORM] = {}
                for var in var_list:
                    if var not in type_compare_dict[event_type][NORM]:
                        type_compare_dict[event_type][NORM][var] = {}
                    for metric in ext_metric_list:                   
                        local_type_data = [type_data[method][event_type][NORM][var][metric] for method in method_list]
                        type_compare_dict[event_type][NORM][var][metric] = get_sorted_method(local_type_data, best="max" if "acc_" in metric else "min")
        for event_type in type_list: 
            for NORM in ["norm", "raw"]:
                for metric in ext_metric_list: 
                    for method, rank in type_compare_dict[event_type][NORM]["all"][metric].items():
                        method_avg_rank_extreme_by_type[method][NORM][metric] += rank / len(type_list)
                
    with open(f"./compare_dict_{FLAG}.json", 'w', encoding='utf-8') as f:
        json.dump({"universal": compare_dict, "type":type_compare_dict}, f, indent=4)
    with open(f"./rank_dict_{FLAG}.json", 'w', encoding='utf-8') as f:
        json.dump({"universal_by_var": method_avg_rank_universal_by_var, "extreme_by_type": method_avg_rank_extreme_by_type}, f, indent=4)

def get_norm_main_table_latex(method2path):
    
    # Load and process the data
    universal_data = {}
    for method, path in method2path.items():
        with open(path, "r") as f:
            df = json.load(f)
            universal_data[method] = df["universal"]["norm"]["all"]
    
    method_list = list(method2path.keys())
    metric_list = list(universal_data[method_list[0]].keys())
    
    # Extract metrics for each method
    method_metrics = {}
    for method in method_list:
        method_metrics[method] = {
            "mse_gen": universal_data[method]["mse_gen"] * 1e3,  # Convert to 1e-3 scale
            "mse_ext": universal_data[method]["mse_ext"] * 1e3,
            "mse_gain": abs(universal_data[method]["mse_gain"] * 1e3),  # Convert to percentage
            "mae_gen": universal_data[method]["mae_gen"] * 1e2,  # Convert to 1e-2 scale
            "mae_ext": universal_data[method]["mae_ext"] * 1e2,
            "mae_gain": abs(universal_data[method]["mae_gain"] * 1e2),
            "rmse_gen": universal_data[method]["rmse_gen"] * 1e2,  # Convert to 1e-2 scale
            "rmse_ext": universal_data[method]["rmse_ext"] * 1e2,
            "rmse_gain": abs(universal_data[method]["rmse_gain"] * 1e2),
            "acc_gen": universal_data[method]["acc_gen"] * 1e1,  # Convert to 1e-1 scale
            "acc_ext": universal_data[method]["acc_ext"] * 1e1,
            "acc_gain": abs(universal_data[method]["acc_gain"] * 1e1),
        }
    
    # Find best and second best values for each metric
    metric_bests = {}
    for metric in ["mae_gen", "mae_ext", "mae_gain",
                    "mse_gen", "mse_ext", "mse_gain", 
                    "rmse_gen", "rmse_ext", "rmse_gain",
                    "acc_gen", "acc_ext", "acc_gain"]:
        values = [method_metrics[method][metric] for method in method_list]
        if "acc" in metric and "gain" not in metric:  # Higher is better for gain and acc metrics
            sorted_values = sorted([(v, i) for i, v in enumerate(values)], reverse=True)
        else:  # Lower is better for other metrics
            sorted_values = sorted([(v, i) for i, v in enumerate(values)])
        
        metric_bests[metric] = {
            "best": sorted_values[0][0],
            "best_method": method_list[sorted_values[0][1]],
            "second": sorted_values[1][0],
            "second_method": method_list[sorted_values[1][1]]
        }
    
    # Generate LaTeX table
    latex_table = inspect.cleandoc(r"""\begin{table*}[t]
        \centering
        \caption{Normalized weather forecasting results (\%). "Gen." and "Ext." indicate "General" and "Extreme".}
        \resizebox{\linewidth}{!}{
        \begin{tabular}{c|ccc|ccc|ccc}
        \toprule
        \multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c|}{\textbf{MAE ($1\times e^{-2}$)}} & \multicolumn{3}{c|}{\textbf{RMSE ($1\times e^{-2}$)}} & \multicolumn{3}{c}{\textbf{ACC ($1\times e^{-1}$)}}\\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} 
        &  \textbf{Ext.$\downarrow$} & \textbf{Gen.$\downarrow$} & \textbf{Gap$\downarrow$} &  \textbf{Ext.$\downarrow$} & \textbf{Gen.$\downarrow$} & \textbf{Gap$\downarrow$} &  \textbf{Ext.$\downarrow$} & \textbf{Gen.$\downarrow$} & \textbf{Gap$\downarrow$}\\
        \hline
        """)
    
    for method in method_list:
        row = f"\\textbf{{{method}}}"
        
        # Add MAE columns
        for m in ["mae_ext", "mae_gen", "mae_gain"]:
            val = method_metrics[method][m]
            if method == metric_bests[m]["best_method"]:
                row += f" & \\textbf{{{val:.4f}}}"
            elif method == metric_bests[m]["second_method"]:
                row += f" & \\underline{{\\textit{{{val:.4f}}}}}"
            else:
                row += f" & {val:.4f}"
                
        # # Add MSE columns
        # for m in ["mse_ext", "mse_gen", "mse_gain"]:
        #     val = method_metrics[method][m]
        #     if method == metric_bests[m]["best_method"]:
        #         row += f" & \\textbf{{{val:.4f}}}"
        #     elif method == metric_bests[m]["second_method"]:
        #         row += f" & \\underline{{\\textit{{{val:.4f}}}}}"
        #     else:
        #         row += f" & {val:.4f}"
        
        # Add RMSE columns
        for m in ["rmse_ext", "rmse_gen", "rmse_gain"]:
            val = method_metrics[method][m]
            if method == metric_bests[m]["best_method"]:
                row += f" & \\textbf{{{val:.4f}}}"
            elif method == metric_bests[m]["second_method"]:
                row += f" & \\underline{{\\textit{{{val:.4f}}}}}"
            else:
                row += f" & {val:.4f}"
        
        # Add ACC columns
        for m in ["acc_ext", "acc_gen", "acc_gain"]:
            val = method_metrics[method][m]
            if method == metric_bests[m]["best_method"]:
                row += f" & \\textbf{{{val:.4f}}}"
            elif method == metric_bests[m]["second_method"]:
                row += f" & \\underline{{\\textit{{{val:.4f}}}}}"
            else:
                row += f" & {val:.4f}"
        
        if method == method_list[-2]:
            row += "\\\\ \n\\hline\\hline\n"
        elif method != method_list[-1]:
            row += "\\\\ \n\\hline\n"
        else:
            row += "\\\\ \n"
        
        latex_table += row
    
    latex_table += inspect.cleandoc(r"""\bottomrule
        \end{tabular}
        }
        \label{table:main_norm}
        \end{table*}""")
    
    # Save to file
    with open(f"norm_main_table.tex", "w") as f:
        f.write(latex_table)
    
    return latex_table

def get_raw_main_table_latex(method2path, metric="mae", used_vars=["msl", "t2m", "z_500", "t_850"]):
    
    # Load and process the data
    universal_data = {}
    for method, path in method2path.items():
        with open(path, "r") as f:
            df = json.load(f)
            universal_data[method] = df["universal"]["raw"]
    
    method_list = list(method2path.keys())
    
    # Process variable names for display
    var_display_names = []
    for var in used_vars:
        # Convert to uppercase and replace underscores
        display_name = var.upper().replace('_', '')
        var_display_names.append(display_name)
    
    # Extract metrics for each method and variable
    method_metrics = {}
    for method in method_list:
        method_metrics[method] = {}
        for var in used_vars:
            method_metrics[method][var] = {
                f"{metric}_gen": universal_data[method][var][f"{metric}_gen"],
                f"{metric}_ext": universal_data[method][var][f"{metric}_ext"],
                f"{metric}_gain": abs(universal_data[method][var][f"{metric}_gain"]),
            }
    
    # Find best and second best values for each metric and variable
    metric_bests = {}
    for var in used_vars:
        metric_bests[var] = {}
        for m in [f"{metric}_ext", f"{metric}_gen", f"{metric}_gain"]:
            values = [method_metrics[method][var][m] for method in method_list]
            if "acc" in metric and "gain" not in m:  # Higher is better for acc metrics
                sorted_values = sorted([(v, i) for i, v in enumerate(values)], reverse=True)
            else:  # Lower is better for other metrics
                sorted_values = sorted([(v, i) for i, v in enumerate(values)])
            
            metric_bests[var][m] = {
                "best": sorted_values[0][0],
                "best_method": method_list[sorted_values[0][1]],
                "second": sorted_values[1][0],
                "second_method": method_list[sorted_values[1][1]]
            }
    
    # Generate the table header parts first
    # Column specification part
    col_spec = "c|" + "|".join(["ccc"] * len(used_vars))
    
    # Multicolumn headers for each variable
    var_headers = "".join([
        f"& \\multicolumn{{3}}{{c|}}{{\\textbf{{{var_display_names[i]}}}}} "
        for i in range(len(used_vars))
    ])
    
    # Cmidrules for the table
    cmidrules = [
        r"\cmidrule(lr){2-4}"
    ] + [
        rf"\cmidrule(lr){{{5+3*i}-{7+3*i}}}"
        for i in range(len(used_vars)-1)
    ]
    cmidrule_str = " ".join(cmidrules)
    
    # Column headers (Gen, Ext, Gap)
    metric_headers = "".join([
        r"& \textbf{Ext.$\downarrow$} & \textbf{Gen.$\downarrow$} & \textbf{Gap$\downarrow$} "
        for _ in used_vars
    ])
    
    # Combine everything into the table header
    latex_header = inspect.cleandoc(rf"""
        \begin{{table*}}[t]
        \centering
        \caption{{Raw-scale weather forecasting MAE results (\%). "Gen." and "Ext." indicate "General" and "Extreme".}}
        \resizebox{{\linewidth}}{{!}}{{ 
        \begin{{tabular}}{{{col_spec}}}
        \toprule
        \multirow{{2}}{{*}}{{\textbf{{Method}}}} {var_headers}\\
        {cmidrule_str}
        {metric_headers}\\
        \hline
        """)
    
    latex_table = latex_header    
    
    for method in method_list:
        row = f"\\textbf{{{method}}}"
        
        # Add columns for each variable
        for var in used_vars:
            for m in [f"{metric}_ext", f"{metric}_gen", f"{metric}_gain"]:
                val = method_metrics[method][var][m]
                if method == metric_bests[var][m]["best_method"]:
                    row += f" & \\textbf{{{val:.4f}}}"
                elif method == metric_bests[var][m]["second_method"]:
                    row += f" & \\underline{{\\textit{{{val:.4f}}}}}"
                else:
                    row += f" & {val:.4f}"
        
        if method == method_list[-2]:
            row += "\\\\ \n\\hline\\hline\n"
        elif method != method_list[-1]:
            row += "\\\\ \n\\hline\n"
        else:
            row += "\\\\ \n"
        
        latex_table += row
    
    latex_table += inspect.cleandoc(r"""\bottomrule
        \end{tabular}
        }
        \label{table:main_raw}
        \end{table*}""")
    
    # Save to file
    with open(f"raw_main_table.tex", "w") as f:
        f.write(latex_table)
    
    return latex_table

def type_specific_plot(method2path, type_to_abbr_2, type_to_abbr_4, metric="mae"):
    # Create output directory
    os.makedirs("type_specific_plots", exist_ok=True)
    
    # Load data
    type_data = {}
    for method, path in method2path.items():
        with open(path, "r") as f:
            df = json.load(f)
            type_data[method] = df["type"]
    
    method_list = list(method2path.keys())
    type_list = list(type_data[method_list[0]].keys())
    
    # Prepare data: {type: {method: metric_ext_value}}
    plot_data = {t: {m: type_data[m][t]["norm"]["all"][f"{metric}_ext"] for m in method_list} 
                 for t in type_list}
    
    # (1) Bar plots - one per event type
    plot_bar_charts(plot_data, method_list, type_list, type_to_abbr_4, metric)
    
    # # (2) Radar chart
    # plot_radar_chart(plot_data, method_list, type_list, type_to_abbr_2, metric)

def type_specific_gain_rank(method2path, type_to_abbr_2, type_to_abbr_4, metric="mae"):
    # Create output directory
    os.makedirs("type_specific_plots", exist_ok=True)
    
    # Load data
    type_data = {}
    for method, path in method2path.items():
        with open(path, "r") as f:
            df = json.load(f)
            type_data[method] = df["type"]
    
    method_list = list(method2path.keys())
    type_list = list(type_data[method_list[0]].keys())
    
    # Prepare data: {type: {method: metric_ext_value}}
    ext_res = {t: {m: type_data[m][t]["norm"]["all"][f"{metric}_ext"] for m in method_list} 
                 for t in type_list}
    
    # Calculate gain percentage for each type and metric
    gain_results = {}
    
    for metric_name, data_dict in [("ext", ext_res)]:
        gain_results[metric_name] = {}
        
        for type_name in type_list:
            
            type_data = data_dict[type_name]
            if "UniExtreme" not in type_data:
                print(f"Warning: UniExtreme not found in {type_name} for {metric_name}")
                raise ValueError
            uniextreme_perf = type_data["UniExtreme"]
            other_methods = {method: perf for method, perf in type_data.items() 
                           if method != "UniExtreme"}
            if not other_methods:
                print(f"Warning: No other methods found in {type_name} for {metric_name}")
                raise ValueError
            
            # Find the best performance among other methods (lower is better)
            best_other_perf = min(other_methods.values())
            best_other_method = [method for method, perf in other_methods.items() 
                               if perf == best_other_perf][0]
            
            # Calculate gain percentage: (other_best - uniextreme) / uniextreme * 100%
            gain_percentage = (best_other_perf - uniextreme_perf) / best_other_perf * 100
            
            gain_results[metric_name][type_name] = {
                "gain_percentage": gain_percentage,
                "uniextreme_performance": uniextreme_perf,
                "best_other_performance": best_other_perf,
                "best_other_method": best_other_method
            }
            
    # Sort gain percentages for each metric
    sorted_gains = {}
    for metric_name in ["ext"]:
        # Get all types and their gain percentages for this metric
        type_gains = [(type_name, gain_results[metric_name][type_name]["gain_percentage"]) 
                        for type_name in gain_results[metric_name]]
        # Sort by gain percentage in descending order (from large to small)
        type_gains.sort(key=lambda x: x[1], reverse=True)
        sorted_gains[metric_name] = type_gains
    print("Sorted gain percentages (from highest to lowest):")
    for metric_name in ["ext"]:
        print(f"\n{metric_name.upper()} metric:")
        for type_name, gain_pct in sorted_gains[metric_name]:
            print(f"  {type_name}: {gain_pct:.2e}%")            
    
    # Save results as JSON with indent
    output_path = "gain_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(gain_results, f, indent=4)
    
    print(f"Gain analysis results saved to {output_path}")
    
    return gain_results   

def plot_bar_charts(plot_data, method_list, type_list, type_to_abbr, metric):
    
    # Define the order of types - first row types first
    first_row_types = ['Flood', 'Waterspout', 'Marine_High_Wind', 'Heavy_Rain', 'Dust_Devil', 
                       'Heat', 'Marine_Strong_Wind', 'Debris_Flow', 'Cold']
    # Get remaining types for second row
    remaining_types = [t for t in type_list if t not in first_row_types]
    
    # Create two rows
    plt.rcParams.update({'font.size': 24})
    fig, axes = plt.subplots(2, 1, figsize=(24, 7))  # Wider figure to accommodate 9 plots per row
    
    cold_colors = ['#a1a9d0', '#b883d4', '#cfeaf1', '#96cccb', '#9e9e9e', '#f7e1ed', '#f0988c']
    colors = cold_colors[:len(method_list)]
    # colors = plt.cm.tab20(np.linspace(0, 1, len(method_list)))
    
    for row_idx, (ax, type_group) in enumerate(zip(axes, [first_row_types, remaining_types])):
        n_cols = len(type_group)
        subaxes = [ax.inset_axes([i/n_cols, 0, 1/n_cols, 1]) for i in range(n_cols)]
        
        # Find global min/max for this row
        row_min = min(min(plot_data[t][m] for m in method_list) for t in type_group)
        row_max = max(max(plot_data[t][m] for m in method_list) for t in type_group)
        padding = (row_max - row_min) * 0.05
        y_lim = (row_min - padding, row_max + padding)
        
        for i, (t, subax) in enumerate(zip(type_group, subaxes)):
            values = [plot_data[t][m] for m in method_list]
            x = np.arange(len(method_list))
            
            bars = subax.bar(x, values, color=colors, edgecolor='black', linewidth=2)
            
            subax.set_title(type_to_abbr.get(t, t[:4]))
            subax.set_xticks([])
            subax.set_ylim(y_lim)
            
            # Only show y-axis for first plot in row
            if i == 0:
                subax.set_ylabel(metric.upper())
                subax.yaxis.set_tick_params(labelsize=16)
            else:
                subax.yaxis.set_visible(False)
        
        # Remove borders and ticks for main axis
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add legend at bottom in a single row
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], label=method, edgecolor='black', linewidth=2) 
                      for i, method in enumerate(method_list)]
    fig.legend(handles=legend_elements, loc='lower center', 
               ncol=len(method_list), bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend
    plt.savefig(os.path.join("type_specific_plots", f"type_bar_{metric}.pdf"), 
                bbox_inches='tight')
    plt.close()

def ablation_plot(method2path, metrics=["mae", "acc"], dim=["gen", "ext", "gain"]):
    
    # Create output directory
    os.makedirs("ablation_plots", exist_ok=True)
    
    # Load data
    universal_data = {}
    for method, path in method2path.items():
        with open(path, "r") as f:
            df = json.load(f)
            universal_data[method] = df["universal"]["norm"]["all"]
    
    method_list = list(method2path.keys())
    
    # Prepare plot data: {metric_dim: {method: value}}
    plot_data = {}
    for metric in metrics:
        for d in dim:
            key = f"{metric}_{d}"
            plot_data[key] = {}
            for method in method_list:
                value = universal_data[method][key]
                # For gain metrics, take absolute value (especially for acc_gain)
                if d == "gain":
                    value = abs(value)
                plot_data[key][method] = value
    
    # Define plot parameters
    plt.rcParams.update({'font.size': 24})
    n_plots = len(metrics) * len(dim)
    fig, ax = plt.subplots(1, n_plots, figsize=(24, 4))
    
    # Use the same colors as in plot_bar_charts
    # cold_colors = ['#a1a9d0', '#b883d4', '#cfeaf1', '#96cccb', '#9e9e9e', '#f7e1ed', '#f0988c']
    cold_colors = ['#e8e0ef', '#c2b1d7', '#c0e2d2', '#afc8e2', '#bfbfbf', '#e2f2cd', '#6fb9d0']
    colors = cold_colors[:len(method_list)]
    
    # Y-axis labels for each subplot
    y_labels = []
    for metric in metrics:
        for d in dim:
            if d == "gen":
                y_labels.append(f"{metric.upper()} Gen.")
            elif d == "ext":
                y_labels.append(f"{metric.upper()} Ext.")
            else:  # gain
                y_labels.append(f"{metric.upper()} Gap")
    
    # Create each subplot
    for i, (key, y_label) in enumerate(zip(plot_data.keys(), y_labels)):
        values = [plot_data[key][m] for m in method_list]
        x = np.arange(len(method_list))
        
        bars = ax[i].bar(x, values, color=colors, edgecolor='black', linewidth=2)
        
        data_range = max(values) - min(values)
        ax[i].set_ylim(min(values) - 0.1*data_range, max(values) + 0.1*data_range)
        
        # Set title and labels
        # ax[i].set_title(y_label)
        ax[i].set_xticks([])
        ax[i].set_ylabel(y_label)
        ax[i].yaxis.set_tick_params(labelsize=16)
        
        # Remove top and right spines
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    
    # Add legend at bottom
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], label=method, edgecolor='black', linewidth=2) 
                      for i, method in enumerate(method_list)]
    fig.legend(handles=legend_elements, loc='lower center', 
               ncol=len(method_list)//2 + 1, bbox_to_anchor=(0.5, -0.15))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    plt.savefig(os.path.join("ablation_plots", f"ablation_{'_'.join(metrics)}.pdf"), 
                bbox_inches='tight')
    plt.close()

def get_raw_all_table_latex_separate(method2path):
    
    # First, load one file to get the list of variables
    sample_path = next(iter(method2path.values()))
    with open(sample_path, "r") as f:
        sample_data = json.load(f)
    variable_list = list(sample_data["universal"]["raw"].keys())
    variable_list.remove("all")
    assert len(variable_list) == 69
    
    # Initialize the output string
    latex_tables = ""
    
    for var_idx, variable in enumerate(variable_list):
        
        if var_idx % 50 == 0:
            latex_tables += inspect.cleandoc(r"""\clearpage
                """) + "\n"
        
        # Load and process the data for this variable
        universal_data = {}
        for method, path in method2path.items():
            with open(path, "r") as f:
                df = json.load(f)
                universal_data[method] = df["universal"]["raw"][variable]
        
        method_list = list(method2path.keys())
        metric_list = list(universal_data[method_list[0]].keys())
        
        # Extract metrics for each method
        method_metrics = {}
        for method in method_list:
            method_metrics[method] = {
                "mse_gen": universal_data[method]["mse_gen"],
                "mse_ext": universal_data[method]["mse_ext"],
                "mse_gain": abs(universal_data[method]["mse_gain"]),
                "mae_gen": universal_data[method]["mae_gen"],
                "mae_ext": universal_data[method]["mae_ext"],
                "mae_gain": abs(universal_data[method]["mae_gain"]),
                "rmse_gen": universal_data[method]["rmse_gen"],
                "rmse_ext": universal_data[method]["rmse_ext"],
                "rmse_gain": abs(universal_data[method]["rmse_gain"]),
                "acc_gen": universal_data[method]["acc_gen"],
                "acc_ext": universal_data[method]["acc_ext"],
                "acc_gain": abs(universal_data[method]["acc_gain"]),
            }
        
        # Find best and second best values for each metric
        metric_bests = {}
        for metric in ["mse_gen", "mse_ext", "mse_gain", 
                       "mae_gen", "mae_ext", "mae_gain",
                       "rmse_gen", "rmse_ext", "rmse_gain",
                       "acc_gen", "acc_ext", "acc_gain"]:
            values = [method_metrics[method][metric] for method in method_list]
            if "acc" in metric and "gain" not in metric:  # Higher is better for acc metrics
                sorted_values = sorted([(v, i) for i, v in enumerate(values)], reverse=True)
            else:  # Lower is better for other metrics
                sorted_values = sorted([(v, i) for i, v in enumerate(values)])
            
            metric_bests[metric] = {
                "best": sorted_values[0][0],
                "best_method": method_list[sorted_values[0][1]],
                "second": sorted_values[1][0],
                "second_method": method_list[sorted_values[1][1]]
            }
        
        # Generate LaTeX table for this variable
        latex_table = inspect.cleandoc(fr"""\begin{{table*}}[ht]
            \centering
            \caption{{Raw weather forecasting results for variable {variable.upper().replace('_', '')}.}}
            \resizebox{{\linewidth}}{{!}}{{
            \begin{{tabular}}{{c|ccc|ccc|ccc}}
            \toprule
            \multirow{{2}}{{*}}{{\textbf{{Method}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{MAE}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{MSE}}}} & \multicolumn{{3}}{{c}}{{\textbf{{ACC}}}}\\
            \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}}
            & \textbf{{Ext.$\downarrow$}} & \textbf{{Gen.$\downarrow$}} & \textbf{{Gap$\downarrow$}} & \textbf{{Ext.$\downarrow$}} & \textbf{{Gen.$\downarrow$}} & \textbf{{Gap$\downarrow$}} & \textbf{{Ext.$\downarrow$}} & \textbf{{Gen.$\downarrow$}} & \textbf{{Gap$\downarrow$}}\\
            \hline
            """)
        
        for method in method_list:
            row = f"\\textbf{{{method}}}"
            
            # Add MSE columns
            for m in ["mse_ext", "mse_gen", "mse_gain"]:
                val = method_metrics[method][m]
                if method == metric_bests[m]["best_method"]:
                    row += f" & \\textbf{{{val:.4e}}}"
                elif method == metric_bests[m]["second_method"]:
                    row += f" & \\underline{{\\textit{{{val:.4e}}}}}"
                else:
                    row += f" & {val:.4e}"
            
            # Add MAE columns
            for m in ["mae_ext", "mae_gen", "mae_gain"]:
                val = method_metrics[method][m]
                if method == metric_bests[m]["best_method"]:
                    row += f" & \\textbf{{{val:.4e}}}"
                elif method == metric_bests[m]["second_method"]:
                    row += f" & \\underline{{\\textit{{{val:.4e}}}}}"
                else:
                    row += f" & {val:.4e}"
            
            # # Add RMSE columns
            # for m in ["rmse_ext", "rmse_gen", "rmse_gain"]:
            #     val = method_metrics[method][m]
            #     if method == metric_bests[m]["best_method"]:
            #         row += f" & \\textbf{{{val:.4e}}}"
            #     elif method == metric_bests[m]["second_method"]:
            #         row += f" & \\underline{{\\textit{{{val:.4e}}}}}"
            #     else:
            #         row += f" & {val:.4e}"
            
            # Add ACC columns
            for m in ["acc_ext", "acc_gen", "acc_gain"]:
                val = method_metrics[method][m]
                if method == metric_bests[m]["best_method"]:
                    row += f" & \\textbf{{{val:.4e}}}"
                elif method == metric_bests[m]["second_method"]:
                    row += f" & \\underline{{\\textit{{{val:.4e}}}}}"
                else:
                    row += f" & {val:.4e}"
            
            if method == method_list[-2]:
                row += "\\\\ \n\\hline\\hline\n"
            elif method != method_list[-1]:
                row += "\\\\ \n\\hline\n"
            else:
                row += "\\\\ \n"
            
            latex_table += row
        
        latex_table += inspect.cleandoc(r"""\bottomrule
            \end{tabular}
            }
            \label{table:raw_""" + variable + r"""}
            \end{table*}""")
        
        # Add this table to the collection with 3 empty lines
        latex_tables += latex_table + "\n\n\n"
    
    # Save all tables to a single file
    with open("raw_all_tables.tex", "w") as f:
        f.write(latex_tables)
    
    return latex_tables

def raw_all_separate_plot(method2path):
    
    def generate_raw_figures_latex_v1(variable_list):
        
        latex_figures = ""
        
        for var_idx, variable in enumerate(variable_list):
            
            if var_idx % 9 == 0:
               latex_figures += inspect.cleandoc(r"""\clearpage
                    """) + "\n"
                
            # Format variable name for display (uppercase, no underscores)
            var_display = variable.upper().replace('_', '')
            # Format filename
            filename = f"raw_metrics_{variable}.pdf"
            
            figure = inspect.cleandoc(fr"""
            \begin{{figure}}[h]
                \centering
                \includegraphics[width=\linewidth]{{figs/main/raw_all_plots/{filename}}}
                \vspace{{-25pt}}
                \caption{{Raw forecasting results of variable {var_display}.}}
                \vspace{{-5pt}}
                \label{{fig:raw_{variable}}}
            \end{{figure}}
            """)
            
            latex_figures += figure + "\n\n"
        
        # Save to file
        with open("raw_all_figures.tex", "w") as f:
            f.write(latex_figures)
        
        return latex_figures
    
    def generate_raw_figures_latex(variable_list):
        latex_figures = ""
        
        for i in range(0, len(variable_list), 4):
            # Start a new page for every 4 figures
            latex_figures += inspect.cleandoc(r"""
                \clearpage
                \begin{figure*}[!ht]
            """) + "\n"
            
            # Process 4 figures (2 per column)
            for j in range(4):
                if i + j >= len(variable_list):
                    break  # Break if no more figures
                
                variable = variable_list[i + j]
                var_display = variable.upper().replace('_', '')
                filename = f"raw_metrics_{variable}.pdf"
                
                # Alternate between left and right minipage
                if j % 2 == 0:
                    latex_figures += inspect.cleandoc(fr"""
                        \begin{{minipage}}[t]{{0.48\textwidth}}
                            \centering
                            \includegraphics[width=\linewidth]{{figs/main/raw_all_plots/{filename}}}
                            \vspace{{-25pt}}
                            \caption{{Raw forecasting results of variable {var_display}.}}
                            \vspace{{-5pt}}
                            \label{{fig:raw_{variable}_left}}
                        \end{{minipage}}
                        \hfill
                    """) + "\n"
                else:
                    latex_figures += inspect.cleandoc(fr"""
                        \begin{{minipage}}[t]{{0.48\textwidth}}
                            \centering
                            \includegraphics[width=\linewidth]{{figs/main/raw_all_plots/{filename}}}
                            \vspace{{-25pt}}
                            \caption{{Raw forecasting results of variable {var_display}.}}
                            \vspace{{-5pt}}
                            \label{{fig:raw_{variable}_right}}
                        \end{{minipage}}
                    """) + "\n"
                    
                    # Add line break after every right minipage (except last)
                    if j < 3 and (i + j + 1) < len(variable_list):
                        latex_figures += inspect.cleandoc(r"""
                            \\[10pt]
                        """) + "\n"
            
            # Close the figure* environment
            latex_figures += inspect.cleandoc(r"""
                \end{figure*}
            """) + "\n\n"
        
        # Save to file
        with open("raw_all_figures.tex", "w") as f:
            f.write(latex_figures)
        
        return latex_figures

    # Create output directory
    os.makedirs("raw_all_plots", exist_ok=True)
    
    # First, load one file to get the list of variables
    sample_path = next(iter(method2path.values()))
    with open(sample_path, "r") as f:
        sample_data = json.load(f)
    variable_list = list(sample_data["universal"]["raw"].keys())
    variable_list.remove("all")
    assert len(variable_list) == 69
    generate_raw_figures_latex(variable_list)
    # exit(-1)
    
    method_list = list(method2path.keys())
    
    # Use the same color scheme as plot_bar_charts
    cold_colors = ['#a1a9d0', '#b883d4', '#cfeaf1', '#96cccb', '#9e9e9e', '#f7e1ed', '#f0988c']
    colors = cold_colors[:len(method_list)]
    
    # Set font size
    plt.rcParams.update({'font.size': 14})
    
    for variable in variable_list:
        # Load and process the data for this variable
        universal_data = {}
        for method, path in method2path.items():
            with open(path, "r") as f:
                df = json.load(f)
                universal_data[method] = df["universal"]["raw"][variable]
        
        # Extract metrics for each method
        method_metrics = {}
        for method in method_list:
            method_metrics[method] = {
                "rmse_gen": universal_data[method]["rmse_gen"],
                "rmse_ext": universal_data[method]["rmse_ext"],
                "rmse_gain": abs(universal_data[method]["rmse_gain"]),
                "mae_gen": universal_data[method]["mae_gen"],
                "mae_ext": universal_data[method]["mae_ext"],
                "mae_gain": abs(universal_data[method]["mae_gain"]),
                "acc_gen": universal_data[method]["acc_gen"],
                "acc_ext": universal_data[method]["acc_ext"],
                "acc_gain": abs(universal_data[method]["acc_gain"]),
            }
        
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(7, 7))
        # fig.suptitle(f"Performance Metrics for Variable {variable.upper().replace('_', '')}", y=1.02)
        
        # Define metric groups and their positions
        metric_groups = [
            ("mae_ext", "mae_gen", "mae_gain"),
            ("rmse_ext", "rmse_gen", "rmse_gain"),
            ("acc_ext", "acc_gen", "acc_gain")
        ]
        row_titles = ["MAE", "RMSE", "ACC"]
        col_titles = ["Extreme", "General", "Gap"]
        
        for row_idx, (row_metrics, row_title) in enumerate(zip(metric_groups, row_titles)):
            for col_idx, metric in enumerate(row_metrics):
                ax = axes[row_idx, col_idx]
                
                # Get values for all methods
                values = [method_metrics[method][metric] for method in method_list]
                x = np.arange(len(method_list))
                
                # Create bars
                bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=2)
                
                # Set titles for first row and first column
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(row_title)
                
                # Adjust y-axis limits with some padding
                data_min, data_max = min(values), max(values)
                padding = (data_max - data_min) * 0.1
                ax.set_ylim(data_min - padding, data_max + padding)
                
                # Remove x-ticks and set scientific notation for y-axis if needed
                ax.set_xticks([])
                ax.yaxis.set_tick_params(labelsize=10)
                
                # # Use scientific notation for MSE metrics
                # if "mse" in metric:
                #     ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Add legend at bottom
        legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], label=method, 
                           edgecolor='black', linewidth=2) 
                          for i, method in enumerate(method_list)]
        # print(method_list)
        fig.legend(handles=legend_elements, loc='lower center', 
                   ncol=3, bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
        
        # Save the figure
        plt.savefig(os.path.join("raw_all_plots", f"raw_metrics_{variable}.pdf"), 
                    bbox_inches='tight')
        plt.close()
        # break

if __name__ == "__main__":
    
    FLAG = "main_new"
    if FLAG == "main":
        method2path = {
            # "NWP": "./Fuxi_down_2_ours/test_logs/logs_nwp_has_type.json",
            "NWP": "./Fuxi_down_2_ours/test_logs/logs_nwp_has_type_sample_miss.json", 
            "GraphCast": "./GraphCast_down_2/test_logs/logs_pretrain_prompt_epoch_33_has_type.json", 
            "Pangu": "./Pangu_down_2/test_logs/logs_pretrain_prompt_best_has_type.json", 
            "FengWu": "./Fengwu_down_2/test_logs/logs_pretrain_prompt_best_has_type.json", 
            "FuXi": "./Fuxi_down_2/test_logs/logs_pretrain_prompt_epoch_15_has_type.json", 
            # "ClimaX": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/ClimaX_down_2/test_logs/logs.json", 
            "OneForecast": "./OneForecast_down_2/test_logs/logs_pretrain_prompt_best_has_type.json", 
            "UniExtreme": "./Fuxi_down_2_ours/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
        }
    elif FLAG == "main_new":
        method2path = {
            "NWP": "./Fuxi_down_2_ours/test_logs/logs_NEW_nwp_has_type.json",
            "GraphCast": "./GraphCast_down_2/test_logs/logs_pretrain_prompt_NEW_epoch_33_has_type.json", 
            "Pangu": "./Pangu_down_2/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            "FengWu": "./Fengwu_down_2/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            "FuXi": "./Fuxi_down_2/test_logs/logs_pretrain_prompt_NEW_epoch_15_has_type.json", 
            "OneForecast": "./OneForecast_down_2/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            "UniExtreme": "./Fuxi_down_2_ours/test_logs/logs_pretrain_prompt_NEW_best_has_type_19-21.json", 
            # "UniExtreme_4": "./Fuxi_down_2_ours/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
        }
    elif FLAG == "ablation":
        method2path = {
            "UniExtreme w/o AFM": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_AFM/test_logs/logs_pretrain_prompt_epoch_16_has_type_19-21.json", 
            "UniExtreme w/o BF": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_BF/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
            "UniExtreme w/o BA": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_BA/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
            "UniExtreme w/o EPA": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_EPA/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
            "UniExtreme w/o MC": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_MC/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
            "UniExtreme w/o MF": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_MF/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
            "UniExtreme": "./Fuxi_down_2_ours/test_logs/logs_pretrain_prompt_best_has_type_19-21.json", 
        }
    elif FLAG == "ablation_new":
        method2path = {
            "UniExtreme w/o AFM": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_AFM/test_logs/logs_pretrain_prompt_NEW_epoch_16_has_type.json", 
            "UniExtreme w/o BF": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_BF/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            "UniExtreme w/o BA": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_BA/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            "UniExtreme w/o EPA": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_EPA/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            # "UniExtreme w/o MC": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_MC/test_logs/logs_pretrain_prompt_NEW_best_has_type.json",
            "UniExtreme w/o MC": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_MC/test_logs/logs_pretrain_prompt_NEW_epoch_24_has_type.json", 
            "UniExtreme w/o MF": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_MF/test_logs/logs_pretrain_prompt_NEW_best_has_type.json", 
            "UniExtreme": "./Fuxi_down_2_ours/test_logs/logs_pretrain_prompt_NEW_best_has_type_19-21.json", 
        }
    else:
        raise ValueError
    
    type_to_abbr_2 = {
        'Flood': 'Fl',
        'Marine_Thunderstorm_Wind': 'MT',
        'Waterspout': 'Ws',
        'Thunderstorm_Wind': 'Tw',
        'Funnel_Cloud': 'Fc',
        'Tornado': 'To',
        'Wind': 'Wd',
        'Hail': 'Hl',
        'Flash_Flood': 'Ff',
        'Lightning': 'Lg',
        'Heavy_Rain': 'Hr',
        'Cold': 'Cd',
        'Marine_High_Wind': 'Mh',
        'Debris_Flow': 'Df',
        'Dust_Devil': 'Dd',
        'Marine_Hail': 'Ml',
        'Heat': 'Ht',
        'Marine_Strong_Wind': 'Ms'
    }
    type_to_abbr_4 = {
        'Flood': 'Flod',
        'Marine_Thunderstorm_Wind': 'MTSW',
        'Waterspout': 'Wtsp',
        'Thunderstorm_Wind': 'TnWn',
        'Funnel_Cloud': 'FnCl',
        'Tornado': 'Trnd',
        'Wind': 'Wind',
        'Hail': 'Hail',
        'Flash_Flood': 'FlFl',
        'Lightning': 'Ltgn',
        'Heavy_Rain': 'HvRn',
        'Cold': 'Cold',
        'Marine_High_Wind': 'MHWn',
        'Debris_Flow': 'DbrF',
        'Dust_Devil': 'DstD',
        'Marine_Hail': 'MrHl',
        'Heat': 'Heat',
        'Marine_Strong_Wind': 'MSWn'
    }
    
    # get_compare_and_rank(method2path)
    method2path = {
        "UniExtreme w/o EPA": "/hpc2hdd/home/hni017/Workplace/ExtremeWeather/Pre-trainedModels/Ablation_wo_EPA/test_logs/logs_pretrain_prompt_NEW_best_has_type.json",
        "UniExtreme": "./Fuxi_down_2_ours/test_logs/logs_pretrain_prompt_NEW_best_has_type_19-21.json", 
    }
    type_specific_gain_rank(method2path, type_to_abbr_2, type_to_abbr_4, metric="mae")
    
    # if FLAG.startswith("main"):
    #     get_norm_main_table_latex(method2path)
    #     get_raw_main_table_latex(method2path, metric="mae", used_vars=["msl", "v_150", "z_500", "t_850"])
        
    #     type_specific_plot(method2path, type_to_abbr_2, type_to_abbr_4, metric="mae")
    #     # type_specific_plot(method2path, type_to_abbr_2, type_to_abbr_4, metric="mse")
    #     type_specific_plot(method2path, type_to_abbr_2, type_to_abbr_4, metric="rmse")
    #     type_specific_plot(method2path, type_to_abbr_2, type_to_abbr_4, metric="acc")
        
    #     raw_all_separate_plot(method2path)
    #     # get_raw_all_table_latex_separate(method2path)
    # elif FLAG.startswith("ablation"):
    #     ablation_plot(method2path, metrics=["mae", "rmse"], dim=["ext", "gen", "gain"])
    # else:
    #     raise ValueError