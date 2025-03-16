// Handle local storage for matches and image data
class localStorageHandler {
    constructor() {
        this.matchesKey = 'leopardMatches';
        this.embeddingsKey = 'leopardEmbeddings';
        this.currentIndexKey = 'currentIndex';
    }

    saveMatches(matches) {
        localStorage.setItem(this.matchesKey, JSON.stringify(matches));
    }

    getMatches() {
        const matches = localStorage.getItem(this.matchesKey);
        return matches ? JSON.parse(matches) : {};
    }

    saveEmbeddings(embeddings) {
        localStorage.setItem(this.embeddingsKey, JSON.stringify(embeddings));
    }

    getEmbeddings() {
        const embeddings = localStorage.getItem(this.embeddingsKey);
        return embeddings ? JSON.parse(embeddings) : null;
    }

    setCurrentIndex(index) {
        localStorage.setItem(this.currentIndexKey, index.toString());
    }

    getCurrentIndex() {
        const index = localStorage.getItem(this.currentIndexKey);
        return index ? parseInt(index) : -1;
    }

    clearAll() {
        localStorage.removeItem(this.matchesKey);
        localStorage.removeItem(this.embeddingsKey);
        localStorage.removeItem(this.currentIndexKey);
    }
}

// Handle file processing
class FileHandler {
    constructor() {
        this.fileReader = new FileReader();
    }

    async readImageAsDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURl(file);
        });
    }

    async loadEmbeddingsFromJson(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                try {
                    const data = JSON.parse(reader.result);
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    async exportMatches(matches) {
        const blob = new Blob([JSON.stringify(matches)], { type: 'application/json' });
        const url = URl.createObjectURl(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'leopard_matches.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URl.revokeObjectURl(url);
    }
}

// Handle image processing and comparison
class ImageProccessor {
    computeSimilarity(embedding1, embedding2) {
        // Implement cosine similarity
        const dotProduct = embedding1.reduce((sum, val, i) => sum + val * embedding2[i], 0);
        const norm1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0));
        const norm2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0));
        return dotProduct / (norm1 * norm2);
    }

    findTopMatches(anchorEmbedding, allEmbeddings, excludeIndices = [], k = 5) {
        const similarities = allEmbeddings
            .map((emb, i) => ({
                index: i,
                similarity: this.computeSimilarity(anchorEmbedding, emb)
            }))
            .filter(item => !excludeIndices.includes(item.index))
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, k);

        return similarities;
    }
}

// Export classes
window.localStorageHandler = localStorageHandler;
window.FileHandler = FileHandler;
window.ImageProccessor = ImageProccessor;
