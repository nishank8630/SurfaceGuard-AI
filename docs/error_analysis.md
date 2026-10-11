# SurfaceGuard AI — Failure Analysis & Limitations

## Failure Modes in Industrial Visual Inspection

1. **Subtle Scratches vs. Surface Reflections (False Negatives)**:
   - *Phenomenon*: Very faint hairline scratches on reflective surfaces (e.g. bottles or polished metal nuts) can blend into specular highlights.
   - *Remedy*: Multi-scale feature extraction combining low-level texture layers (`layer1`/`layer2`) with directional morphological filters.

2. **Edge Misalignments & Boundary High Scores (False Positives)**:
   - *Phenomenon*: Slight natural rotation, translation, or lighting variation near product boundaries can trigger high patch distances.
   - *Remedy*: Boundary masking, spatial jitter augmentation during training, or increasing Gaussian smoothing radius.

3. **Reconstruction Blur in Autoencoders**:
   - *Phenomenon*: Standard L2/L1 convolutional autoencoders tend to output slightly blurred reconstructions around high-frequency texture transitions. Normal sharp edges may produce residual false positives.
   - *Remedy*: PatchCore-style feature retrieval operates directly in intermediate representation space, avoiding pixel reconstruction blur entirely.

## Portfolio & Research Scope Notice

SurfaceGuard AI is an academic and portfolio research implementation intended to demonstrate visual anomaly detection principles. It does not carry ISO/CE industrial safety certification and should not be used as the sole automated fail-safe in hazardous industrial machinery without physical interlocks.
