# A Comparative Study of Relative and Metric Monocular Depth Estimation for Video-Based Indoor 3D Reconstruction

**Author:** Rax  
**Affiliation:** Department of Computer Vision and 3D Reconstruction, Indian Institute of Technology (Indian School of Mines), Dhanbad  
**Date:** September 2026  
**Document:** [depth-estimation-comparison.docx](depth-estimation-comparison.docx)

---

## Abstract

This paper examines two monocular depth estimation pipelines that convert a short, hand-captured panoramic video of an indoor room into a coloured three-dimensional point cloud. Both pipelines share an identical surrounding framework: frame extraction with FFmpeg, panoramic mosaicking with OpenCV, spherical back-projection of per-pixel depth into Cartesian coordinates, statistical outlier removal, and export to the Polygon File Format (PLY). The two implementations differ only in the depth-prediction network employed. 

The first, referred to here as the **Depth Anything V2** pipeline, relies on a relative depth-estimation transformer and recovers an approximate notion of scale through a fixed, scene-independent rescaling heuristic. The second, the **ZoeDepth** pipeline, employs a network trained explicitly to output metric depth through a dedicated bins-based decoding stage. We describe the internal architecture of each depth model, document the surrounding processing pipeline stage by stage, and compare the two systems on the dimension that matters most for downstream use: whether the reconstructed geometry corresponds to real-world measurements. 

We find that while both models produce visually coherent point clouds, only the ZoeDepth pipeline yields output whose scale can be trusted, and we argue this makes it the more appropriate choice for any application in which measurement, rather than appearance, is the objective.

**Keywords:** monocular depth estimation, ZoeDepth, Depth Anything V2, vision transformers, image stitching, point cloud reconstruction, metric depth, panoramic 3D reconstruction.

---

## 1. Introduction

Recovering three-dimensional structure from ordinary video remains a central problem in computer vision, with applications spanning indoor mapping, real estate visualisation, robotic navigation, and augmented reality. Classical multi-view approaches such as structure-from-motion (SfM) recover metrically correct geometry but require dense camera coverage and are sensitive to textureless surfaces common in indoor scenes (e.g., painted walls, ceilings, uniform floors). An alternative that has become practical only in the last several years is single-image monocular depth estimation using deep neural networks, which sidesteps the need for multi-view correspondence entirely.

Two families of monocular depth networks exist:
1. **Relative depth estimation networks** are trained across large, heterogeneous image collections and generalise well to unfamiliar scenes, but their output is correct only up to an unknown scale and shift.
2. **Metric depth estimation networks** are trained on a narrower set of datasets that provide ground-truth real-world depth (e.g., RGB-D sensors), and consequently output values that can be interpreted in physical units (metres), at some cost to cross-domain generalisation.

This paper studies one representative of each family — **Depth Anything V2** and **ZoeDepth** — as they are deployed inside otherwise identical video-to-point-cloud pipelines, and asks which is the more suitable choice for constructing a point cloud that is not merely plausible-looking but dimensionally meaningful.

The remainder of this paper is organised as follows: Section 2 reviews background concepts common to both pipelines. Section 3 describes the shared processing pipeline. Section 4 documents the Depth Anything V2 implementation and its scale-recovery heuristic. Section 5 documents the ZoeDepth model architecture and implementation. Section 6 compares the two systems and argues for ZoeDepth as the stronger choice. Section 7 concludes.

---

## 2. Background

The distinction between relative and metric depth estimation is central to this study. Relative depth networks are typically trained with a scale-and-shift-invariant loss across many datasets spanning indoor, outdoor, and synthetic imagery, which produces depth maps that are structurally correct — nearer and farther surfaces are ordered properly — but cannot be read as metres or any other physical unit without additional information. Metric depth networks are instead trained, or fine-tuned, on datasets that provide calibrated ground-truth depth, such as NYU Depth v2 for indoor scenes, and therefore learn to output values in real units, though typically at the cost of weaker generalisation outside the training domain.

Both networks examined here build on the **Dense Prediction Transformer (DPT)** design, in which an image is divided into patches, encoded by a vision transformer at several depths, reassembled into multi-resolution feature maps, and progressively fused by a convolutional decoder into a dense, full-resolution prediction. This architecture forms the shared foundation on which the two pipelines' depth models are built, and the difference between them lies chiefly in what is attached after this common backbone.

The surrounding pipeline in both notebooks additionally depends on classical, non-learned computer vision techniques:
- **Panoramic stitching**, performed with OpenCV's `Stitcher` class, detects local keypoints in each extracted frame, matches them across overlapping frames, estimates the relative homographies between cameras, performs bundle adjustment to align all frames globally, and warps and blends the frames onto a common projection surface to produce a single wide image.
- **Point Cloud Representation**, stored in the Polygon File Format (PLY), a simple container developed at Stanford University for point and mesh data together with per-vertex colour, chosen here for its broad compatibility with tools such as MeshLab, CloudCompare, and Blender.

---

## 3. Shared Pipeline Architecture

Both notebooks implement the same eleven-stage, single-shot pipeline:
1. Dependencies are installed.
2. Frames are extracted from an input video with FFmpeg at a fixed sampling rate (e.g., 2 fps).
3. A depth model is loaded into GPU memory.
4. Extracted frames are stitched into a panoramic mosaic using OpenCV.
5. Black boundary borders resulting from homography warps are cropped using contour analysis.
6. The stitched panorama is passed through the depth network.
7. Bilateral filtering is applied to edge-preserve and smooth depth noise.
8. The resulting depth map is back-projected to 3D coordinates under a camera model.
9. Spurious noise points are removed using a KDTree statistical outlier filter.
10. The cleaned, coloured point cloud is exported to PLY format.
11. An interactive 3D preview is rendered.

The pipeline is *single-shot* in the sense that depth is inferred once, from the stitched panorama, rather than per frame with subsequent multi-view fusion; this avoids the need for explicit camera pose estimation and expensive bundling, at the cost of treating the stitched panorama as though it were captured from a single sweeping viewpoint with a fixed horizontal field of view. It is against this common scaffold that the two depth models are best compared, since any difference in the reconstructed geometry can be attributed to the depth-prediction stage rather than to the surrounding processing.

---

## 4. The Depth Anything V2 Pipeline

### 4.1 Model
The Depth Anything V2 pipeline loads its depth model through a single Hugging Face Transformers call, instantiating the Small variant of the network:
```python
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
```
Depth Anything V2 follows the same encoder-decoder transformer design described in Section 2, using a distilled vision transformer encoder trained across a very large corpus of real and synthetically generated images with a scale-and-shift-invariant objective. Its output, however, terminates at the relative-depth stage: the network was never fine-tuned with any metric head against ground-truth real-world depth, and its predicted values are properly interpreted as an inverse-depth, or disparity-like, ordering rather than a measurement.

### 4.2 Implementation and Scale Recovery
Following frame extraction and panoramic stitching, which proceed identically to the shared pipeline described above, the notebook obtains a raw disparity tensor from the model and converts it into a depth-like quantity through a short sequence of operations: the disparity is inverted, the result is clipped to its fifth and ninety-fifth percentile to suppress extreme values, the clipped range is normalised to the interval $[0, 1]$, and the normalised map is finally rescaled into a fixed span of 1 to 6 units before a bilateral filter is applied for edge-preserving smoothing:

$$\text{depth} = \text{clip}\left(\frac{1}{\text{disparity} + \epsilon}, p_5, p_{95}\right)$$
$$\text{depth}_{\text{norm}} = \frac{\text{depth} - p_5}{p_{95} - p_5 + \epsilon}$$
$$\text{depth}_{\text{scaled}} = 5 \cdot \text{depth}_{\text{norm}} + 1$$

The pixels are subsequently back-projected into three-dimensional coordinates using a single horizontal viewing angle $\theta$ derived from a fixed $140^\circ$ field of view, together with a vertical offset proportional to depth and inverse focal length:

$$X = d \sin(\theta), \quad Z = d \cos(\theta), \quad Y = -\left(v - \frac{h}{2}\right) \cdot \frac{d}{f_x}$$

The resulting coloured points are filtered with a $k$-d tree-based statistical outlier removal before being written to an ASCII PLY file.

### 4.3 Limitations of the Scale-Recovery Heuristic
The rescaling procedure in Section 4.2 does not recover metric scale in any principled sense. Because every processed scene is clipped to its own percentile range and then linearly stretched into the same fixed one-to-six-unit interval, a small room and a considerably larger room would, after this normalisation, occupy an identical numeric span; the transformation is scene-relative rather than scene-invariant, and its output units bear no fixed relationship to metres. The resulting point cloud can be visually convincing, since the underlying relative-depth ordering from the network is generally reliable, but any absolute distance, wall length, or object dimension read from the exported PLY file should be treated as illustrative rather than measured.

---

## 5. The ZoeDepth Pipeline

### 5.1 Model Architecture
The ZoeDepth pipeline loads its depth model through PyTorch Hub:
```python
zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True).to(device).eval()
```
Its architecture extends the relative-depth backbone with three additional components specifically designed to produce metric output:
1. **Metric Bins Module:** Attached to the pre-trained decoder; rather than regressing a single depth value directly, this module discretises the plausible depth range for the scene into a set of learned bins, in the manner of an adaptive histogram, and predicts an initial estimate of the bin centres from the decoder's bottleneck features.
2. **Attractor Layers:** Operate at each of the four subsequent decoder resolutions. At every level, a small multilayer perceptron predicts a set of attractor points in depth space, and the current bin centres are pulled toward these points through a bounded update rule rather than being discarded and re-predicted. This allows the globally consistent coarse bin layout to be sharpened locally as finer spatial detail becomes available without destabilising the depth ordering learned during pre-training.
3. **Log-Binomial Decoding:** The final per-pixel depth is computed not by an unordered softmax over the bins but by the expectation of a log-binomial distribution over the attractor-refined bin centres, a choice that architecturally encodes the fact that depth bins are ordered physical quantities.

Three official ZoeDepth configurations exist:
- `ZoeD_N`: Fine-tuned on the indoor **NYU Depth v2** dataset.
- `ZoeD_K`: Fine-tuned on the outdoor **KITTI** driving dataset.
- `ZoeD_NK`: Combines both heads behind an automatic domain router.

The indoor room pipeline utilizes `ZoeD_N`, which matches the indoor nature of the room-scanning task and avoids any risk of router misclassification. At inference, the loaded model exposes `infer_pil`, which executes the full forward pass through the backbone, metric bins module, and attractor-refined log-binomial decoder, returning a per-pixel depth map expressed directly in **metres**.

### 5.2 Implementation and Observed Results
Frame extraction and panoramic stitching produce, in the benchmark run, 26 frames combined into a panorama of $1774 \times 712$ pixels after border cropping. The stitched panorama is passed once through `infer_pil` to obtain a metric depth map, which is then smoothed with a bilateral filter ($d=9, \sigma_{\text{color}}=0.5, \sigma_{\text{space}}=15$).

Back-projection uses spherical-to-Cartesian projection with independent horizontal and vertical viewing angles ($\theta$ and $\phi$) derived from a $120^\circ$ horizontal FOV:

$$\theta = \left(\frac{u}{W} - 0.5\right) \cdot \text{FOV}_x$$
$$\phi = \left(\frac{v}{H} - 0.5\right) \cdot \text{FOV}_y$$
$$X = d \sin(\theta) \cos(\phi)$$
$$Y = -d \sin(\phi)$$
$$Z = d \cos(\theta) \cos(\phi)$$

Statistical outlier removal is implemented with a $k$-d tree over each point's 30 nearest neighbours ($k=31$ including self) and a threshold of $\mu + 1.5\sigma$. In the test run:
- Initial cloud: **315,772 points**
- Filtered cloud: **298,642 points** (94.57% retained, ~5.43% filtered as noise)
- File format: Binary PLY with vertex coordinates and RGB colors

Because the depth values entering this back-projection are themselves metric, the resulting $X, Y, Z$ coordinates correspond to physical distances in metres.

---

## 6. Comparative Analysis

The two pipelines are architecturally identical outside their depth-prediction stage, which isolates the comparison to a single variable: whether the network's output is relative or metric.

### Table 1: Detailed Architectural and Empirical Comparison

| Property | Depth Anything V2 (ViT-S) | ZoeDepth (ZoeD_N) |
| :--- | :--- | :--- |
| **Output Quantity** | Inverse disparity (relative, unit-free) | Metric depth in **metres** |
| **Scale Source** | Percentile clip & fixed linear remap $[1.0, 6.0]$ | Learned metric bins module fine-tuned on NYUv2 |
| **Depth Ordering Constraint** | None beyond base regression loss | Log-binomial expectation enforces ordered bins |
| **Cross-Scene Consistency** | Not guaranteed; scenes squashed to same range | **Consistent physical scale** across diverse rooms |
| **Projection Geometry** | Cylindrical coordinate mapping | Spherical $(\theta, \phi)$ back-projection |
| **Outlier Rejection** | $k=21$, threshold $\mu + 2.0\sigma$ | $k=31$, threshold $\mu + 1.5\sigma$ |
| **Target Applications** | Relative-depth effects, visual previews, rendering | **Robotics, CAD/BIM, floor plans, real measurement** |

The practical consequence of this difference concerns what the exported point cloud can be used for. A cloud built from Depth Anything V2's rescaled disparity retains the correct relative arrangement of surfaces, but its absolute scale is an artefact of the fixed 1-to-6 normalisation rather than a property of the scene. A cloud built from ZoeDepth's metric output, by contrast, inherits the physical grounding of NYU Depth v2: distances between points correspond to real distances in metres.

For applications such as robotic path planning, real-estate floor-area estimation, or AR furniture placement, ZoeDepth provides a decisive advantage. Depth Anything V2 remains a lightweight choice where only qualitative 3D relief or rendering preview is required.

---

## 7. Conclusion

This paper has documented two video-to-point-cloud pipelines differing in their choice of monocular depth network. Depth Anything V2 provides strong relative depth but requires an external, scene-independent heuristic to approximate scale. ZoeDepth extends the transformer backbone with a metric bins module and attractor-based refinement trained explicitly to produce depth in real units. For applications in which the reconstructed geometry must be measured, rather than merely viewed, ZoeDepth is the decisively more appropriate model.

---

## References

1. **Bhat, S. F., Birkl, R., Wofk, D., Wonka, P., & Müller, M. (2023).** *ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth.* arXiv:2302.12288.
2. **Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021).** *Vision Transformers for Dense Prediction.* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
3. **Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024).** *Depth Anything V2.* arXiv:2406.09414.
4. **Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012).** *Indoor Segmentation and Support Inference from RGBD Images.* NYU Depth v2 dataset. European Conference on Computer Vision (ECCV).
5. **Brown, M., & Lowe, D. G. (2007).** *Automatic Panoramic Image Stitching using Invariant Features.* International Journal of Computer Vision (IJCV), 74(1), 59–73.
