# author: David Colomer Matachana
# GitHub username: acse-dc1823
from flask import Flask, jsonify, request, render_template, send_file
import numpy as np
import os
import shutil
import csv
import networkx as nx
import json
from waitress import serve
import sys

import logging

app = Flask(__name__)

# Global variables
GLOBAL_MATCH_DIR = ""
FINAL_OUTPUT_DIR = "final_leopards"
graph = nx.Graph()
embeddings = None
distance_matrix = None
image_filenames = []
binary_image_filenames = []
CURRENT_DB = "leopard_matches.json"
last_processed_index = -1
EMBEDDINGS_PATH = ""
UNCROPPED_IMAGES_PATH = ""

def load_embeddings(embeddings_folder):
    """Load embeddings and related data from the specified folder."""
    global embeddings, distance_matrix, image_filenames, binary_image_filenames, EMBEDDINGS_PATH
    EMBEDDINGS_PATH = embeddings_folder
    embeddings = np.load(os.path.join(embeddings_folder, "embeddings.npy"))
    distance_matrix = np.load(os.path.join(embeddings_folder, "distance_matrix.npy"))

    with open(os.path.join(embeddings_folder, "image_filenames.txt"), "r") as file:
        image_filenames = [line.strip() for line in file]
    print(len(image_filenames), "images loaded")

    with open(os.path.join(embeddings_folder, "binary_image_filenames.txt"), "r") as file:
        binary_image_filenames = [line.strip() for line in file]
    print(len(binary_image_filenames), "binary images loaded")

def load_or_create_db(db_name):
    """Load an existing database or create a new one if it doesn't exist."""
    global graph, CURRENT_DB, last_processed_index
    CURRENT_DB = db_name if db_name.endswith(".json") else db_name + ".json"
    graph.clear()

    if os.path.exists(CURRENT_DB):
        # Load existing graph
        with open(CURRENT_DB, "r") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data["graph"])
        last_processed_index = data.get("last_processed_index", -1)
        action = "loaded"
    else:
        # Create new graph
        last_processed_index = -1
        action = "created"

    # Ensure all image paths are added as nodes
    for image_path in image_filenames:
        if image_path not in graph:
            graph.add_node(image_path)

    save_db()
    return action

def save_db():
    """Save the current graph and last processed index to the database file."""
    data = {"graph": nx.node_link_data(graph), "last_processed_index": last_processed_index}
    with open(CURRENT_DB, "w") as f:
        json.dump(data, f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_match_dir", methods=["POST"])
def set_match_dir():
    """Set the global match directory."""
    global GLOBAL_MATCH_DIR
    GLOBAL_MATCH_DIR = request.json["match_dir"]
    if not os.path.exists(GLOBAL_MATCH_DIR):
        os.makedirs(GLOBAL_MATCH_DIR)
    return jsonify({"status": "success", "message": f"Directory set to {GLOBAL_MATCH_DIR}"})

@app.route("/open_existing_embeddings", methods=["POST"])
def open_existing_embeddings():
    """Load existing embeddings from a specified path."""
    embeddings_path = request.json["embeddings_path"]
    if not os.path.exists(embeddings_path):
        return jsonify({"status": "error", "message": "Embeddings folder does not exist"})
    load_embeddings(embeddings_path)
    return jsonify({"status": "success", "message": "Embeddings loaded successfully"})

@app.route("/run_model_from_scratch", methods=["POST"])
def run_model_from_scratch():
    """Run the inference model to create embeddings from scratch using specified folders."""
    print("Running model from scratch")
    output_folder = request.json["output_folder"]
    unprocessed_image_folder = request.json["unprocessed_image_folder"]

    temp_config_path = os.path.join(os.path.dirname(__file__), "temp_config_inference.json")
    print("Temp config path:", temp_config_path)

    try:
        # Get the absolute path to the config file for both development and bundled contexts
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))
        print("Base path:", base_path)
        
        config_path = os.path.abspath(
            os.path.join(base_path, "leopard_id", "config_inference.json")
        )
        print("Config path:", config_path)
        
        # Load the base config and update with new paths
        with open(config_path, "r") as f:
            config = json.load(f)
            print("successfully loaded config")
            config["output_folder"] = output_folder
            config["unprocessed_image_folder"] = unprocessed_image_folder
        
        # Save updated config to temporary file and ensure it's properly written
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
            print("successfully saved temp config")
        
        # Import and run inference
        from leopard_id.inference_embeddings import run_inference
        print("successfully imported run_inference")
        run_inference(temp_config_path)
        
        # Load the newly created embeddings
        load_embeddings(output_folder)
        return jsonify({"status": "success", "message": "Model run successfully and embeddings loaded"})

    except Exception as e:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        return jsonify({"status": "error", "message": f"Error running inference: {str(e)}"})

    finally:
        # Clean up temp file if it exists
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

@app.route("/set_uncropped_images_path", methods=["POST"])
def set_uncropped_images_path():
    """Set the path for uncropped images."""
    global UNCROPPED_IMAGES_PATH
    UNCROPPED_IMAGES_PATH = request.json["uncropped_images_path"]
    if not os.path.exists(UNCROPPED_IMAGES_PATH):
        return jsonify({"status": "error", "message": "Uncropped images folder does not exist"})
    return jsonify({"status": "success", "message": "Uncropped images path set successfully"})

@app.route("/create_or_open_db", methods=["POST"])
def create_or_open_db():
    """Create a new database or open an existing one."""
    db_name = request.json["db_name"]
    action = load_or_create_db(db_name)
    return jsonify(
        {"status": "success", "message": f"Database {CURRENT_DB} {action} and set as current"}
    )

def distance_to_confidence(distance):
    """
    Maps a distance value between two images to a confidence score between 0 and 100.
    The confidence score is calculated as 100 * exp(-2 * (distance - 0.45)).
    This, if the distance is closer than 0.45 (for cosine distance, this means
    it being arccos(0.55)=56.6 degrees or closer), the confidence score is 100.
    """
    score = 100 * np.exp(-2 * (max(distance - 0.45, 0)))
    return score

@app.route("/validate_match", methods=["POST"])
def validate_match():
    """Validate a match between two images."""
    data = request.json
    print("Received data in validate_match:", data)

    if data["is_match"]:
        add_match(data["anchor"], data["match"])

    return jsonify({"status": "success"})

@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve an image file to front end."""
    full_path = next((f for f in image_filenames if os.path.basename(f) == filename), None)
    if full_path and os.path.exists(full_path):
        return send_file(full_path)
    return "File not found", 404

@app.route("/binary_images/<path:filename>")
def serve_binary_image(filename):
    """Serve a binary image file to front end."""
    full_path = next((f for f in binary_image_filenames if os.path.basename(f) == filename), None)
    if full_path and os.path.exists(full_path):
        return send_file(full_path)
    return "File not found", 404

@app.route("/get_next_anchor", methods=["GET"])
def get_next_anchor():
    """Get the index of the next unprocessed anchor image."""
    global last_processed_index
    current_index = int(request.args.get("current_index", last_processed_index))
    next_index = current_index + 1

    # Find the next unprocessed anchor
    while next_index < len(image_filenames):
        if next_index > last_processed_index:
            break
        next_index += 1

    # If we've reached the end, start over from the beginning
    if next_index >= len(image_filenames):
        next_index = 0
        while next_index <= last_processed_index:
            if not list(graph.neighbors(image_filenames[next_index])):
                break
            next_index += 1

        # If we've cycled through all images and found no unprocessed ones, keep the current index
        if next_index > last_processed_index:
            next_index = current_index

    last_processed_index = max(last_processed_index, next_index)
    save_db()

    # If we've cycled through all images, organize the final output
    if next_index == 0 and current_index == len(image_filenames) - 1:
        organize_final_output()

    return jsonify({"next_index": next_index})

@app.route("/get_anchor_and_similar", methods=["GET"])
def get_anchor_and_similar():
    """Get the anchor image and similar images for comparison."""
    global last_processed_index
    anchor_index = int(request.args.get("index", last_processed_index))
    anchor_path = image_filenames[anchor_index]
    binary_anchor_path = binary_image_filenames[anchor_index]

    # Ensure the anchor path is in the graph
    if anchor_path not in graph:
        graph.add_node(anchor_path)

    # Get all images already matched with the anchor (directly or indirectly)
    matched_images = set()
    for component in nx.connected_components(graph):
        if anchor_path in component:
            matched_images = component
            break

    # Get distances for all images
    distances = distance_matrix[anchor_index]

    # Create a list of (index, distance) tuples, excluding matched images and the anchor itself
    distance_list = [
        (i, dist)
        for i, dist in enumerate(distances)
        if image_filenames[i] not in matched_images and i != anchor_index
    ]

    # Sort by distance and take top 5
    sorted_distances = sorted(distance_list, key=lambda x: x[1])[:5]

    # Extract indices and distances
    indices, distances = zip(*sorted_distances) if sorted_distances else ([], [])

    similar_images = ["/images/" + os.path.basename(image_filenames[i]) for i in indices]
    confidences = [distance_to_confidence(d) for d in distances]

    last_processed_index = anchor_index
    save_db()

    # Get information about the current connected component
    current_component = list(matched_images)
    component_size = len(current_component)

    return jsonify(
        {
            "anchor": "/images/" + os.path.basename(anchor_path),
            "binary_anchor": "/binary_images/" + os.path.basename(binary_anchor_path),
            "anchor_path": anchor_path,
            "similar": similar_images,
            "binary_similar": [
                "/binary_images/" + os.path.basename(image_filenames[i]) for i in indices
            ],
            "similar_paths": [image_filenames[i] for i in indices],
            "anchor_index": anchor_index,
            "similar_indices": list(indices),
            "confidences": confidences,
            "component_size": component_size,
            "component_paths": ["/images/" + os.path.basename(path) for path in current_component],
        }
    )

def organize_final_output():
    """Organize the final output by copying matched images to their respective directories."""
    if not GLOBAL_MATCH_DIR:
        return "Match directory not set", 400
    # Get all connected components (each component represents a unique leopard)
    connected_components = list(nx.connected_components(graph))

    # Prepare data for CSV and organize images
    csv_data = []
    for i, component in enumerate(connected_components, 1):
        leopard_id = f"leopard_{i}"
        leopard_dir = os.path.join(GLOBAL_MATCH_DIR, leopard_id)
        os.makedirs(leopard_dir, exist_ok=True)

        for image_path in component:
            # Extract only the filename without any directory information
            image_filename = os.path.basename(image_path)
            dest_path = os.path.join(leopard_dir, image_filename)

            # Copy the file to the new location
            shutil.copy2(image_path, dest_path)

            # Add the original path and leopard ID to the CSV data
            csv_data.append([image_path, leopard_id])

    # Write to CSV
    csv_filename = os.path.join(GLOBAL_MATCH_DIR, "leopard_matches.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Path", "Leopard ID"])  # Header
        writer.writerows(csv_data)

    return f"Final output organized in {GLOBAL_MATCH_DIR}", 200

@app.route("/end_session", methods=["POST"])
def end_session():
    """End the current session and organize the final output."""
    print("Ending session")
    message, status_code = organize_final_output()
    return jsonify({"status": "success" if status_code == 200 else "error", "message": message})

def add_match(anchor, match):
    """Add a match between two images in the graph."""
    global last_processed_index
    anchor_path = image_filenames[anchor]
    match_path = image_filenames[match]

    # Add edge between anchor and match if it doesn't exist
    if not graph.has_edge(anchor_path, match_path):
        graph.add_edge(anchor_path, match_path)
        print(f"Added edge between {anchor_path} and {match_path}")
    else:
        print(f"Edge already exists between {anchor_path} and {match_path}")

    last_processed_index = max(last_processed_index, anchor)
    save_db()
    print_graph_state()

def print_graph_state():
    """Print the current state of the graph for debugging purposes."""
    print("\nCurrent Graph State:")
    for component in nx.connected_components(graph):
        if len(component) > 1:
            print(f"Connected component with {len(component)} nodes:")
            for node in component:
                print(f"  - {node}")
    print(f"Total number of nodes: {graph.number_of_nodes()}")
    print(f"Total number of edges: {graph.number_of_edges()}")
    print(f"Number of connected components: {nx.number_connected_components(graph)}")

@app.route("/debug_graph", methods=["GET"])
def debug_graph():
    """
    Return debug information about the current state of the graph,
    including the number of nodes, edges, connected components.
    """
    components = list(nx.connected_components(graph))
    component_sizes = [len(c) for c in components]
    largest_component = max(components, key=len)

    return jsonify(
        {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "number_of_components": len(components),
            "component_sizes": component_sizes,
            "largest_component_size": len(largest_component),
            "largest_component_nodes": list(largest_component),
        }
    )

if __name__ == "__main__":
    load_or_create_db(CURRENT_DB)
    serve(app, host="127.0.0.1", port=5000, threads=2)
