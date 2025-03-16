# Active Context: SpotID

## Current Status

### Model Performance
- DT5AP: 0.8814
- T5RMD: 0.9533
- Successfully outperforms Triplet Network baseline
- Slightly underperforms compared to Hotspotter

### Active Components
1. Model Architecture
   - Modified CosFace implementation
   - Adaptive angular margin function
   - EfficientNetV2-B2 backbone

2. Preprocessing Pipeline
   - YOLO-based detection working
   - Background removal has 3.2% failure rate
   - Edge detection has 7.7% partial extraction rate
   - Combined RGB + edge detection channel approach proving effective

3. Interface
   - Flask-based implementation stable
   - Graph database effectively tracking matches
   - Progress saving/loading functional

## Current Challenges

### Technical Challenges
1. Background Removal
   - 3.2% of usable leopard flanks not accurately identified
   - 5.2% of flanks only partially extracted

2. Edge Detection
   - 7.7% of images have partially extracted patterns
   - Particularly problematic with uneven illumination

3. Model Limitations
   - Struggles with high occlusion rates
   - Limited by dataset size
   - Angular margin could be further optimized

### User Experience Challenges
1. Processing Speed
   - Preprocessing pipeline relatively slow (10-15s per image)
   - Need for faster background removal

2. Interface
   - Requires local installation
   - No cloud storage option yet
   - Limited multi-user support

## Active Decisions

### Recently Implemented
1. Modified CosFace margin function
   ```python
   h(θ) = ((1-cos²(x))/4 + 0.1)/1.1
   ```

2. Combined channel approach
   - RGB + edge detection channels
   - Improved performance over single channel

3. Graph database for match tracking
   - Efficient storage of relationships
   - Prevents redundant comparisons

### Under Consideration
1. Improved background removal
   - Investigating newer segmentation models
   - Potential for custom trained model

2. Web-based deployment
   - Cloud storage integration
   - Multi-user support
   - Progress synchronization

3. Performance optimization
   - Batch processing improvements
   - Caching strategies
   - Parallel processing

## Next Steps

### Immediate Actions
1. Optimize preprocessing pipeline
   - Investigate faster background removal options
   - Implement parallel processing
   - Add preprocessing caching

2. Enhance interface
   - Add cloud storage support
   - Implement multi-user features
   - Add batch processing option
   - Add ease of use for non technical users, no need to use any code.

### Future Directions
1. Web Application Development
   - Design cloud architecture
   - Implement user authentication
   - Add collaborative features

2. Model Improvements
   - Research newer architectures
   - Experiment with larger models
   - Investigate multi-task learning

3. Species Expansion
   - Test on other big cats
   - Adapt preprocessing pipeline
   - Fine-tune model architecture

## Active Metrics to Monitor
1. Processing Time
   - Preprocessing duration
   - Model inference speed
   - Interface response time

2. Accuracy Metrics
   - DT5AP trends
   - T5RMD performance
   - False positive rates

3. User Experience
   - Session duration
   - Match verification speed
   - Error occurrence rates
