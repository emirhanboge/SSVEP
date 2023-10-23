import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
import warnings
import sys
import matplotlib.cm as cm

# TODO: CHeck if there are negatives

warnings.filterwarnings("ignore")

SNR_THRESHOLD = 1.5

def extract_timestamp_and_layer(filename):
    parts = filename.split("_")
    if len(parts) < 4:
        return None, None

    timestamp = int(parts[2])
    layer = parts[3].replace(".npy", "").replace("layer", "")

    try:
        layer_num = int(layer)
    except ValueError:
        return None, None

    return timestamp, layer_num

def sorted_filenames_from_folder(folder_name):
    all_files = os.listdir(folder_name)
    valid_files = [f for f in all_files if "pre" not in f and f.endswith('.npy') and extract_timestamp_and_layer(f)[0] is not None]
    valid_files.sort(key=lambda x: extract_timestamp_and_layer(x))
    uniq_timestamps = set([extract_timestamp_and_layer(f)[0] for f in valid_files])
    return valid_files, uniq_timestamps

def load_sorted_activations(folder_name): # 120 (images) * 34 (layers)
    sorted_files, uniq_timestamps = sorted_filenames_from_folder(folder_name)
    activations = {}
    img_count = 0
    counter = 0
    for filename in sorted_files:
        timestamp, layer_name = extract_timestamp_and_layer(filename)
        if img_count not in activations:
            activations[img_count] = {}
        if layer_name not in activations[img_count]:
            activations[img_count][layer_name] = {}
        activations[img_count][layer_name] = np.load(os.path.join(folder_name, filename))
        counter += 1
        if counter == 34:
            counter = 0
            img_count += 1

    nodes_values = {}
    for img in activations:
        for layer in activations[img]:
            if nodes_values == {} or layer not in nodes_values.keys():
                nodes_values[layer] = {}
            curr_act_shape = activations[img][layer].shape
            node_count = curr_act_shape[1]
            for node in range(node_count):
                if nodes_values[layer] == {} or node not in nodes_values[layer].keys():
                    nodes_values[layer][node] = []
                if layer != 34:
                    squeezed = np.squeeze(activations[img][layer])
                    node_act = squeezed[node]
                    mean_of_node = np.mean(node_act)
                    nodes_values[layer][node].append(mean_of_node)
                elif layer == 34:
                    nodes_values[layer][node].append(activations[img][layer][0][node])

    return nodes_values, len(uniq_timestamps)

def calculate_snr(fft_values, bins, skip):
    snr_values = []
    for i in range(len(fft_values)):
        if i < len(fft_values) - (bins + skip):
            right_noise = fft_values[i+skip:i+skip+bins+1]
        else:
            right_noise = fft_values[i+skip:len(fft_values)]
        if i > skip + bins:
            left_noise = fft_values[i-skip-bins:i-skip+1]
        else:
            left_noise = fft_values[0:max(i-skip, 0)]
        noise_baseline = np.append(left_noise, right_noise)
        baseline_avg = np.average(noise_baseline)
        if baseline_avg == 0:
            snr = 0
        else:
            snr = fft_values[i] / baseline_avg
        snr_values.append(snr)
    return snr_values

def calculate_snr_for_node(grads, bins=3, skip=1):
    # Grads is 120 values for a node in a particular layer
    fft_output = np.fft.rfft(grads)
    fft_output = np.abs(fft_output)
    fft_output = fft_output[1:]
    fft_output = np.abs(fft_output)
    snr_values_for_node = calculate_snr(fft_output, bins, skip)
    snr_val = np.mean(snr_values_for_node)
    return snr_values_for_node, snr_val

def is_high_snr(node_data):
    snr_values_for_node, snr = calculate_snr_for_node(node_data)
    return snr > SNR_THRESHOLD, snr, snr_values_for_node

def get_snr_nodes(layer_data):
    # nodes: {Node_{k}: {is_high: True/False, snr: 1.5}}
    # interesting_nodes: {Node_{k}: 1.5}
    nodes = {}
    interesting_nodes = {}
    snr_values_for_nodes = {}
    snr_values_for_interesting_nodes = {}
    for node in layer_data:
        is_high, snr, snr_values_for_node = is_high_snr(layer_data[node])

        nodes[node] = {
            "is_high": is_high,
            "snr": snr
        }

        if is_high:
            interesting_nodes[node] = snr

        snr_values_for_nodes[node] = snr_values_for_node
        snr_values_for_interesting_nodes = [snr_values_for_nodes[node] for node in interesting_nodes]

    number_of_interesting_nodes = len(list(interesting_nodes.keys()))

    return interesting_nodes, nodes, number_of_interesting_nodes, snr_values_for_nodes, snr_values_for_interesting_nodes

def plot_layer_evolution(layer_data, layer, folder_name):
    interesting_nodes, nodes, number_of_interesting_nodes, snr_values_for_nodes, snr_values_for_interesting_nodes = get_snr_nodes(layer_data)

    interesting_nodes_names = list(interesting_nodes.keys())
    total_nodes = len(list(layer_data.keys()))
    percentage_interesting_fl = len(interesting_nodes_names) / total_nodes * 100

    fig, axs = plt.subplots(1, 2, figsize=(14, 18))
    title_info_fl = (f'Layer {layer} | Total Nodes: {total_nodes} | Interesting Nodes: {len(interesting_nodes_names)} | '
                     f'Percentage Interesting: {percentage_interesting_fl:.2f}% | '
                     f'SNR Threshold: {SNR_THRESHOLD}')
    title_info_fl_all = (f'Showing all Nodes')

    fig.suptitle(title_info_fl, fontsize=14)

    colors_for_nodes = {}
    for node in layer_data.keys():
        colors_for_nodes[node] = cm.jet(node / len(layer_data.keys()))

    # Plot for interesting nodes
    for idx, node in enumerate(interesting_nodes_names):
        axs[0].plot(snr_values_for_interesting_nodes[idx], label=f'Node {node}', color=colors_for_nodes[node])
    axs[0].set(title=f'SNR (Interesting Nodes)', ylabel='SNR')
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

    # Plot for all nodes
    for node in snr_values_for_nodes:
        axs[1].plot(snr_values_for_nodes[node], label=f'Node {node}', color=colors_for_nodes[node])
    axs[1].set(title=f'SNR (All Nodes)', ylabel='SNR')

    output_dir = f"{folder_name}/layer_evolution"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plt.savefig(os.path.join(output_dir, f"layer_{layer}.png"))
    plt.close()
    return total_nodes, len(interesting_nodes_names), interesting_nodes_names

def are_activations_same(activations1, activations2, rtol=1e-5, atol=1e-8):
    for img in activations1:
        for layer in activations1[img]:
            for node in activations1[img][layer]:
                if not np.allclose(activations1[img][layer][node], activations2[img][layer][node], rtol=rtol, atol=atol):
                    return False
    return True

def analysis(flicker_folder):
    nodes_values, img_count = load_sorted_activations(flicker_folder)

    # nodes_values: {Layer_{j}: {Node_{k}: [120 values]}}
    layers = list(nodes_values.keys())
    totals = []
    interestings = []
    interesting_nodes = {} # {Layer_{j}: [Node_{k}]}
    for layer in layers:
        if layer == 34:
            continue
        layer_data = nodes_values[layer]
        total, interesting, interesting_nodes_layer = plot_layer_evolution(layer_data, layer, flicker_folder)
        totals.append(total)
        interestings.append(interesting)
        interesting_nodes[layer] = interesting_nodes_layer

    filename = f"{flicker_folder}/layer_evolution/interesting_nodes.json"
    with open(filename, 'w') as outfile:
        json.dump(interesting_nodes, outfile)

    filename = f"{flicker_folder}/layer_evolution/interesting_nodes_counts.json"
    with open(filename, 'w') as outfile:
        text = f"Total Nodes: {sum(totals)}\n" \
                f"Total Interesting Nodes: {sum(interestings)}\n" \
                f"Percentage Interesting Nodes: {sum(interestings) / sum(totals) * 100:.2f}%\n\n" \
                f"Interesting Nodes Counts:\n"
        outfile.write(text)
        for layer in interesting_nodes:
            outfile.write(f"Layer {layer}: {len(interesting_nodes[layer])}\n")

if __name__ == '__main__':
    flicker_folders = os.listdir("pruning_experiments")
    analysis_results = {}
    for flicker_folder in flicker_folders:
        if len(os.listdir(f"pruning_experiments/{flicker_folder}")) != 120*34:
            continue
        interesting_nodes_layer = analysis(flicker_folder)
        analysis_results[flicker_folder] = interesting_nodes_layer

    common_interesting_nodes = {}
    for flicker_folder in analysis_results:
        for layer in analysis_results[flicker_folder]:
            if layer not in common_interesting_nodes:
                common_interesting_nodes[layer] = set(analysis_results[flicker_folder][layer])
            else:
                common_interesting_nodes[layer] = common_interesting_nodes[layer].intersection(set(analysis_results[flicker_folder][layer]))

    with open("pruning_experiments/common_interesting_nodes.json", 'w') as outfile:
        json.dump(common_interesting_nodes, outfile, indent=4)

    node_seen_count = {}
    for flicker_folder in analysis_results:
        for layer in analysis_results[flicker_folder]:
            for node in analysis_results[flicker_folder][layer]:
                if layer not in node_seen_count:
                    node_seen_count[layer] = {}
                if node not in node_seen_count[layer]:
                    node_seen_count[layer][node] = 1
                else:
                    node_seen_count[layer][node] += 1

    with open("pruning_experiments/node_seen_count.json", 'w') as outfile:
        json.dump(node_seen_count, outfile, indent=4)






