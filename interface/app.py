from flask import Flask, jsonify, request, render_template, send_file
import numpy as np
import sqlite3
import os
import shutil

app = Flask(__name__)

# Global variables
GLOBAL_MATCH_DIR = ""
CURRENT_DB = "leopard_matches.db"
embeddings = np.load('../leopard_id/embeddings/embeddings.npy')
distance_matrix = np.load('../leopard_id/embeddings/distance_matrix.npy')
image_filenames = []
with open('../leopard_id/embeddings/image_filenames.txt', 'r') as file:
    image_filenames = [line.strip() for line in file]

def init_db(db_name='leopard_matches.db'):
    global CURRENT_DB
    CURRENT_DB = db_name
    conn = sqlite3.connect(CURRENT_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS leopards
                 (leopard_id TEXT PRIMARY KEY)''')
    c.execute('''CREATE TABLE IF NOT EXISTS matches
                 (anchor INTEGER, match INTEGER, is_match BOOLEAN, leopard_id TEXT,
                 FOREIGN KEY(leopard_id) REFERENCES leopards(leopard_id))''')
    conn.commit()
    conn.close()

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

def distance_to_confidence(distance, max_distance=50):
    scale = max_distance / 3
    score = 100 * np.exp(-distance / scale)
    print(distance, score)
    return score

@app.route('/get_anchor_and_similar', methods=['GET'])
def get_anchor_and_similar():
    anchor_index = int(request.args.get('index', 0))
    distances = distance_matrix[anchor_index]
    indices = np.argsort(distances)
    indices = indices[1:6]
    similar_images = ["/images/" + os.path.basename(image_filenames[i]) for i in indices]
    confidences = [distance_to_confidence(distances[i]) for i in indices]
    return jsonify({
        'anchor': "/images/" + os.path.basename(image_filenames[anchor_index]),
        'anchor_path': image_filenames[anchor_index],
        'similar': similar_images,
        'similar_paths': [image_filenames[i] for i in indices],
        'anchor_index': anchor_index,
        'similar_indices': indices.tolist(),
        'confidences': confidences
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    full_path = next((f for f in image_filenames if os.path.basename(f) == filename), None)
    if full_path and os.path.exists(full_path):
        return send_file(full_path)
    return "File not found", 404

def get_or_create_leopard_id(conn, anchor, match):
    c = conn.cursor()
    
    # Check if either anchor or match is already associated with a leopard
    c.execute("SELECT DISTINCT leopard_id FROM matches WHERE anchor IN (?, ?) OR match IN (?, ?)",
              (anchor, match, anchor, match))
    existing_ids = c.fetchall()
    
    if existing_ids:
        # If multiple IDs found, we need to merge them
        leopard_id = existing_ids[0][0]
        if len(existing_ids) > 1:
            for old_id in existing_ids[1:]:
                c.execute("UPDATE matches SET leopard_id = ? WHERE leopard_id = ?",
                          (leopard_id, old_id[0]))
                c.execute("DELETE FROM leopards WHERE leopard_id = ?", (old_id[0],))
        return leopard_id
    else:
        # Create new leopard ID
        leopard_id = f"leopard_{anchor}_{match}"
        c.execute("INSERT INTO leopards (leopard_id) VALUES (?)", (leopard_id,))
        return leopard_id

@app.route('/validate_match', methods=['POST'])
def validate_match():
    data = request.json
    conn = sqlite3.connect(CURRENT_DB)
    
    leopard_id = None
    if data['is_match']:
        leopard_id = get_or_create_leopard_id(conn, data['anchor'], data['match'])
        
        if GLOBAL_MATCH_DIR:
            leopard_dir = os.path.join(GLOBAL_MATCH_DIR, leopard_id)
            os.makedirs(leopard_dir, exist_ok=True)
            
            # Copy matched images to the leopard directory if they're not already there
            for img_index in (data['anchor'], data['match']):
                img_filename = os.path.basename(image_filenames[img_index])
                dest_path = os.path.join(leopard_dir, img_filename)
                if not os.path.exists(dest_path):
                    shutil.copy2(image_filenames[img_index], dest_path)

    c = conn.cursor()
    c.execute("INSERT INTO matches (anchor, match, is_match, leopard_id) VALUES (?, ?, ?, ?)",
              (data['anchor'], data['match'], data['is_match'], leopard_id))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'leopard_id': leopard_id})

@app.route('/get_next_anchor', methods=['GET'])
def get_next_anchor():
    current_index = int(request.args.get('current_index', -1))
    next_index = (current_index + 1) % len(image_filenames)
    return jsonify({'next_index': next_index})

@app.route('/create_new_db', methods=['POST'])
def create_new_db():
    db_name = request.json['db_name']
    init_db(db_name)
    return jsonify({'status': 'success', 'message': f"New database {db_name} created and set as current"})

@app.route('/open_existing_db', methods=['POST'])
def open_existing_db():
    global CURRENT_DB
    db_name = request.json['db_name']
    if os.path.exists(db_name):
        CURRENT_DB = db_name
        return jsonify({'status': 'success', 'message': f"Database {db_name} opened and set as current"})
    else:
        return jsonify({'status': 'error', 'message': 'Database not found'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)