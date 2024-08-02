from flask import Flask, jsonify, request, render_template, send_file
import numpy as np
import os
import shutil
import csv
import networkx as nx
import json

app = Flask(__name__)

# Global variables
GLOBAL_MATCH_DIR = ""
FINAL_OUTPUT_DIR = "final_leopards"
graph = nx.Graph()
embeddings = np.load('../leopard_id/embeddings/embeddings.npy')
distance_matrix = np.load('../leopard_id/embeddings/distance_matrix.npy')
image_filenames = []
with open('../leopard_id/embeddings/image_filenames.txt', 'r') as file:
    image_filenames = [line.strip() for line in file]
    print(len(image_filenames), "images loaded")

CURRENT_DB = "leopard_matches.json"
last_anchor_index = 0

def load_or_create_db(db_name):
    global graph, CURRENT_DB, last_anchor_index
    CURRENT_DB = db_name if db_name.endswith('.json') else db_name + '.json'
    graph.clear()
    
    if os.path.exists(CURRENT_DB):
        # Load existing graph
        with open(CURRENT_DB, 'r') as f:
            data = json.load(f)
        graph = nx.node_link_graph(data['graph'])
        last_anchor_index = data.get('last_anchor_index', 0)
        action = "loaded"
    else:
        # Create new graph
        last_anchor_index = 0
        action = "created"
    
    # Ensure all image paths are added as nodes
    for image_path in image_filenames:
        if image_path not in graph:
            graph.add_node(image_path)
    
    save_db()
    return action

def save_db():
    data = {
        'graph': nx.node_link_data(graph),
        'last_anchor_index': last_anchor_index
    }
    with open(CURRENT_DB, 'w') as f:
        json.dump(data, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/set_match_dir', methods=['POST'])
def set_match_dir():
    global GLOBAL_MATCH_DIR
    GLOBAL_MATCH_DIR = request.json['match_dir']
    if not os.path.exists(GLOBAL_MATCH_DIR):
        os.makedirs(GLOBAL_MATCH_DIR)
    return jsonify({'status': 'success', 'message': f"Directory set to {GLOBAL_MATCH_DIR}"})

@app.route('/create_or_open_db', methods=['POST'])
def create_or_open_db():
    db_name = request.json['db_name']
    action = load_or_create_db(db_name)
    return jsonify({'status': 'success', 'message': f"Database {CURRENT_DB} {action} and set as current"})

def distance_to_confidence(distance, max_distance=50):
    scale = max_distance / 3
    score = 100 * np.exp(-distance / scale)
    return score

@app.route('/validate_match', methods=['POST'])
def validate_match():
    data = request.json
    print("Received data in validate_match:", data)
    
    if data['is_match']:
        add_match(data['anchor'], data['match'])
    
    return jsonify({'status': 'success'})


@app.route('/images/<path:filename>')
def serve_image(filename):
    full_path = next((f for f in image_filenames if os.path.basename(f) == filename), None)
    if full_path and os.path.exists(full_path):
        return send_file(full_path)
    return "File not found", 404

@app.route('/get_next_anchor', methods=['GET'])
def get_next_anchor():
    global last_anchor_index
    print("Last anchor index:", last_anchor_index)
    current_index = int(request.args.get('current_index', last_anchor_index))
    next_index = current_index + 1
    
    # Find the next unmatched anchor
    while next_index < len(image_filenames):
        if not list(graph.neighbors(image_filenames[next_index])):
            break
        next_index += 1
    
    # If we've reached the end, start over from the beginning
    if next_index >= len(image_filenames):
        next_index = 0
        while next_index < current_index:
            if not list(graph.neighbors(image_filenames[next_index])):
                break
            next_index += 1
        
        # If we've cycled through all images and found no unmatched ones, keep the current index
        if next_index >= current_index:
            next_index = current_index
    
    last_anchor_index = next_index
    save_db()
    
    # If we've cycled through all images, organize the final output
    if next_index == 0:
        organize_final_output()
    
    return jsonify({'next_index': next_index})

def organize_final_output():
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
    csv_filename = os.path.join(GLOBAL_MATCH_DIR, 'leopard_matches.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Path', 'Leopard ID'])  # Header
        writer.writerows(csv_data)

    return f"Final output organized in {GLOBAL_MATCH_DIR}", 200

@app.route('/end_session', methods=['POST'])
def end_session():
    print("Ending session")
    message, status_code = organize_final_output()
    return jsonify({'status': 'success' if status_code == 200 else 'error', 'message': message})

def add_match(anchor, match):
    global last_anchor_index
    anchor_path = image_filenames[anchor]
    match_path = image_filenames[match]
    
    # Add edge between anchor and match if it doesn't exist
    if not graph.has_edge(anchor_path, match_path):
        graph.add_edge(anchor_path, match_path)
        print(f"Added edge between {anchor_path} and {match_path}")
    else:
        print(f"Edge already exists between {anchor_path} and {match_path}")
    
    last_anchor_index = max(last_anchor_index, anchor)
    save_db()
    print_graph_state()

def print_graph_state():
    print("\nCurrent Graph State:")
    for component in nx.connected_components(graph):
        if len(component) > 1:
            print(f"Connected component with {len(component)} nodes:")
            for node in component:
                print(f"  - {node}")
    print(f"Total number of nodes: {graph.number_of_nodes()}")
    print(f"Total number of edges: {graph.number_of_edges()}")
    print(f"Number of connected components: {nx.number_connected_components(graph)}")

@app.route('/get_anchor_and_similar', methods=['GET'])
def get_anchor_and_similar():
    global last_anchor_index
    anchor_index = int(request.args.get('index', last_anchor_index))
    anchor_path = image_filenames[anchor_index]
    
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
    distance_list = [(i, dist) for i, dist in enumerate(distances) 
                     if image_filenames[i] not in matched_images and i != anchor_index]
    
    # Sort by distance and take top 5
    sorted_distances = sorted(distance_list, key=lambda x: x[1])[:5]
    
    # Extract indices and distances
    indices, distances = zip(*sorted_distances) if sorted_distances else ([], [])
    
    similar_images = ["/images/" + os.path.basename(image_filenames[i]) for i in indices]
    confidences = [distance_to_confidence(d) for d in distances]
    
    last_anchor_index = anchor_index
    save_db()
    
    # Get information about the current connected component
    current_component = list(matched_images)
    component_size = len(current_component)
    
    return jsonify({
        'anchor': "/images/" + os.path.basename(anchor_path),
        'anchor_path': anchor_path,
        'similar': similar_images,
        'similar_paths': [image_filenames[i] for i in indices],
        'anchor_index': anchor_index,
        'similar_indices': list(indices),
        'confidences': confidences,
        'component_size': component_size,
        'component_paths': ["/images/" + os.path.basename(path) for path in current_component]
    })

@app.route('/debug_graph', methods=['GET'])
def debug_graph():
    components = list(nx.connected_components(graph))
    component_sizes = [len(c) for c in components]
    largest_component = max(components, key=len)
    
    return jsonify({
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'number_of_components': len(components),
        'component_sizes': component_sizes,
        'largest_component_size': len(largest_component),
        'largest_component_nodes': list(largest_component)
    })

if __name__ == '__main__':
    load_or_create_db(CURRENT_DB)
    app.run(debug=True, port=5000)