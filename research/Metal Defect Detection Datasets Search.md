# **Exhaustive Analysis of Public Domain Datasets for Industrial Metal Surface Defect Detection**

## **Introduction to Industrial Metallurgy and Automated Optical Inspection**

The transition toward Industry 4.0 and the paradigm of smart manufacturing has fundamentally transformed the principles of quality assurance in modern metallurgy and industrial production pipelines.1 At the core of this transformation is the deployment of automated optical inspection technologies, which leverage advanced machine vision to autonomously identify, localize, and classify physical anomalies on the surfaces of manufactured goods.2 In the specific context of rolled metal products—encompassing hot-rolled steel strips, flat sheet steel, extruded aluminum profiles, and continuous copper strips—surface integrity is not merely an aesthetic concern but a foundational parameter of material performance. Physical defects such as micro-cracks, deep scratches, and imbedded oxides can severely compromise the mechanical strength of the metal, significantly reduce its durability under cyclic loading, and precipitate catastrophic structural failures in downstream applications ranging from aerospace engineering and automotive manufacturing to heavy infrastructure.4

Historically, the detection of surface defects in rolling mills relied heavily on human visual inspection, a methodology that is inherently fraught with inefficiencies, high latency, and subjective inconsistency.5 The sheer velocity of modern production lines, where hot-rolled strips can travel at speeds exceeding ten meters per second, makes manual inspection impossible to scale effectively.6 The initial advent of traditional machine vision sought to address this by introducing algorithmic approaches utilizing artificially designed features and conventional image processing techniques, such as edge detection and contrast thresholding.2 However, these traditional deterministic methods exhibited significant limitations regarding versatility and computational robustness. They struggled continuously to adapt to the highly dynamic optical environments of rolling mills, where fluctuating illumination, varying thermal gradients, and the highly reflective properties of metallic surfaces induce severe signal noise and trigger high false-positive rates.3

The integration of deep learning, particularly the utilization of Convolutional Neural Networks and modern Vision Transformers, has catalyzed a massive paradigm shift in defect detection capabilities.5 These deep architectural models excel at autonomously extracting hierarchical, non-linear feature representations directly from raw pixel data, bypassing the need for brittle, hand-crafted feature engineering. Yet, the functional efficacy and deployment viability of any deep learning model are intrinsically tethered to the quality, scale, representational fidelity, and precise annotation of its underlying training datasets.9 For a computer vision model to successfully traverse the chasm from a highly controlled academic laboratory environment to the chaotic reality of a real-world production line, it must be trained on datasets that accurately mirror the statistical and optical realities of industrial manufacturing.

A critical, yet historically overlooked, requirement for these industrial datasets is the inclusion of "normal" or entirely defect-free images.10 In a real-world continuous casting or hot-rolling facility, the overwhelming majority of the produced material is pristine and defect-free. If a neural network is trained exclusively on an artificially curated dataset containing solely defective images, it implicitly learns a distorted prior probability that every ingested frame contains an anomaly. This leads directly to a prohibitively high false-positive rate upon deployment, rendering the automated system useless as it flags acceptable material for scrap. Furthermore, the precise spatial localization of the anomaly is equally as important as its categorical classification. Industrial robotic mitigation systems, quality control dashboards, and automated sorting mechanisms require exact spatial coordinates to excise, track, or monitor the damaged portion of the metal strip. Therefore, datasets must provide exact bounding box annotations—or fine-grained instance segmentation masks that can be programmatically converted to bounding boxes—alongside full-image categorical labels to facilitate multi-task object detection algorithms.12

This comprehensive research report provides an exhaustive, critical analysis of publicly available, open-source datasets tailored for rolled metal surface defect detection. It specifically targets repositories that fulfill the rigorous demands of modern algorithmic research and real-world industrial deployment. These criteria include the mandatory presence of negative (defect-free) samples, the provision of precise bounding box spatial annotations combined with whole-image categorical labels, the presence of highly specific metallurgical defect class taxonomies, and unrestricted public access through widely used platforms such as Kaggle and GitHub, eliminating the barrier of proprietary paywalls.12

## **The Dual Imperative of Bounding Boxes and Image-Level Classification**

The architectural demands of modern automated optical inspection transcend the simple boundaries of basic image classification. To fulfill the operational requirements of a rolling mill, the artificial intelligence system must answer three distinct questions simultaneously: "What is the defect?", "Where is the defect?", and "How many defects are present?".2 This necessitates a dataset that provides a dual-annotation paradigm, supplying both bounding boxes that localize the damage and overarching image-level labels that categorize the specific metallurgical failure.

The simultaneous provision of bounding boxes and image-level classification labels enables the training of advanced multi-task learning networks. When an entire image is labeled with a specific defect class, it provides the network with global contextual semantic information. The network learns to associate the macroscopic texture, lighting conditions, and overall grain structure of the metal sheet with the specific category of damage. Simultaneously, the bounding box provides highly localized, micro-semantic data, forcing the convolutional layers to identify the exact pixel gradients and edge geometries that separate the defect from the surrounding healthy metal matrix.

This dual-labeling approach is critical for single-stage detectors like the You Only Look Once network architectures. These models divide the input image into a grid and assign bounding boxes and class predictions to each grid cell simultaneously.15 The network locates and identifies the objects by predicting bounding box coordinates and the corresponding probabilities of the classes linked to those boxes.15 If a dataset provides only bounding boxes without a rigorous, specific defect classification for the image, the network acts merely as a region proposal tool, unable to inform the engineering team whether the localized anomaly is a harmless water spot or a catastrophic structural crack. Conversely, if the dataset provides only an image-level label without bounding boxes, the network cannot guide downstream mitigation equipment to the physical location of the damage. Thus, the intersection of bounding box localization and specific categorical classification is the mandatory foundation of any viable industrial dataset.

## **The Mathematical and Operational Necessity of Negative Samples**

To fully grasp the utility of the datasets analyzed in this report, it is imperative to examine the underlying mathematical, statistical, and algorithmic principles that govern object detection in actual industrial environments. The most profound discrepancy between legacy academic datasets and modern real-world datasets is the distribution of the data itself.

In the realm of physical quality control, the distribution of data is inherently pathological and heavily skewed. A dataset derived from an optimized manufacturing plant will be overwhelmingly dominated by non-defective, pristine samples, leading to what is algorithmically known as a severe foreground-background class imbalance.16 Conversely, datasets historically utilized in early academic research often consisted of perfectly balanced subsets of strictly defective images.17

Training an object detection model or a two-stage detector on an exclusively defective dataset artificially skews the network's internal confidence thresholds.15 When such a poorly trained model encounters a complex, highly textured background pattern in production—such as a benign fluid mark, a non-structural machining line, or a reflection from an overhead stroboscopic illuminant—it attempts to force a classification based on the defect features it has memorized, lacking the knowledge of what a "normal" complex surface looks like.

The intentional introduction of negative, defect-free samples into the training pipeline allows the neural network to calculate a substantially more accurate loss gradient during the backpropagation phase.9 A network must be mathematically penalized for drawing a bounding box around healthy metal. By utilizing advanced optimization techniques such as Hard Negative Mining or Focal Loss architectures, the neural network explicitly learns to penalize these false positives aggressively.9 Focal loss, specifically, is designed to reshape the standard cross-entropy loss function such that it exponentially down-weights the loss assigned to easily classified, standard examples, forcing the model's gradient descent to concentrate almost entirely on hard, misclassified examples.16 In industrial settings, these "hard" examples are frequently normal images with high-variance textures or confusing optical reflections. Thus, datasets containing a substantial ratio of normal images are not merely an operational preference; they are a fundamental, mathematical prerequisite for training robust, production-ready defect discrimination algorithms.11

## **Exhaustive Review of Real-World Imbalanced Datasets**

The following repositories represent the vanguard of publicly available data for metal surface defect detection. They have been explicitly selected based on their strict adherence to the required criteria: containing high volumes of negative samples, providing bounding box capabilities or convertible segmentation masks, categorizing defects meticulously, and being hosted on open-source platforms like Kaggle.

### **The Severstal Steel Defect Detection Dataset**

The Severstal Steel Defect Detection dataset, officially released in 2019 by the Russian metallurgical company PAO Severstal, stands as one of the most comprehensive, challenging, and realistic public datasets available to the machine vision community.12 Hosted prominently on the Kaggle platform as part of a global algorithmic competition, the dataset was designed explicitly to aid engineers and researchers in refining algorithms for detecting, localizing, and categorizing surface defects on flat sheet steel directly during the production process.12

#### **Dataset Composition and Real-World Fidelity**

The production process of flat sheet steel is highly delicate; from the initial heating and hot-rolling phases to the final drying and cutting stages, numerous heavy machines sequentially interact with the flat steel.12 Each interaction creates a unique opportunity for surface degradation. To capture this reality, Severstal utilized specialized high-frequency cameras operating directly above the active production line to drive their defect detection algorithms.12 The resulting dataset is monumental in scale and fidelity, containing a total of 18,074 high-resolution images, with spatial dimensions standardized to 1600 pixels in width and 256 pixels in height.12

Crucially, the Severstal dataset accurately reflects the statistical rarity of defects in a highly optimized manufacturing environment, satisfying the imperative need for non-defective data. Of the 18,074 total images, exactly 11,408 are completely unlabeled, representing entirely defect-free, normal images.12 This constitutes a massive 63% of the total dataset, providing an unparalleled repository of negative samples necessary for training robust background-discrimination capabilities and minimizing false-positive generation.12 The remaining 37% of the images contain at least one labeled defect, yielding a total of 19,958 distinct labeled objects scattered across the defective subset.12

#### **Annotation Formats and Defect Distribution**

The dataset is annotated natively using pixel-level instance segmentation.12 While instance segmentation provides an exact pixel-wise mask of the defect, it is highly computationally intensive to train on. However, due to the nature of instance segmentation masks, the data can be automatically and cleanly transformed into object detection tasks by algorithmically drawing bounding boxes around the extremum coordinates of every individual masked object.12 The defects themselves are categorized into four distinct classes, each exhibiting unique spatial distributions and frequencies 12:

| Severstal Defect Class | Image Count | Total Labeled Objects | Average Objects per Image | Average Area Coverage |
| :---- | :---- | :---- | :---- | :---- |
| **Defect\_3** | 5,150 | 14,648 | 2.84 | 6.22% (Max 80.53%) |
| **Defect\_1** | 897 | 3,082 | 3.44 | 1.06% |
| **Defect\_4** | 801 | 1,907 | 2.38 | 8.39% (Max 41.05%) |
| **Defect\_2** | 247 | 321 | 1.30 | 0.82% |

This dataset inherently poses the highly realistic challenge of defect multiplicity; a single image frame may contain zero defects, a singular defect of one class, or a highly complex, heterogeneous mix of multiple defect classes overlapping one another.12 Due to its high volume of normal images, rigorous image-level specific defect classifications, and free public availability via Kaggle, the Severstal dataset is arguably the premier asset for developing real-world bounding box detection algorithms for flat rolled steel.

### **The Kolektor Surface-Defect Dataset 2 (KolektorSDD2)**

While the Severstal dataset models the production of wide flat sheet steel, the Kolektor Surface-Defect Dataset 2 (KolektorSDD2) models the production of specific industrial metallic items and electrical commutators.21 Provided by the Visual Cognitive Systems Laboratory at the University of Ljubljana in conjunction with the industrial partner Kolektor Group d.o.o., KolektorSDD2 is an exemplary resource for modeling extreme, almost pathological class imbalance in real-world manufacturing environments.21

#### **Extreme Imbalance Metrics**

Captured with a dedicated visual inspection system in a highly controlled industrial environment, the KolektorSDD2 dataset contains 3,336 images with spatial dimensions of approximately 230 pixels in width and 630 pixels in height.21 The dataset pushes the ratio of normal to defective images to the extreme edge of the spectrum, providing an incredibly challenging benchmark for background suppression.

In total, 2,980 images out of the 3,336 are completely normal, unlabeled images.21 This means that a staggering 89% of the dataset consists of negative samples.21 The dataset is meticulously bifurcated into predefined training and testing splits to ensure standardized benchmarking 21:

* **Training Set:** Contains 2,085 negative (defect-free) samples juxtaposed against only 246 positive (defective) samples.  
* **Test Set:** Contains 894 negative samples and a mere 110 positive samples.

The dataset features a generalized "defect" class that encompasses various morphological anomalies observed on the surface of the item, ranging from microscopic scratches and minor spots to macroscopic surface imperfections and structural fractures.21

#### **Architectural Adaptability**

Similar to the Severstal dataset, KolektorSDD2 natively provides fine-grained, pixel-level instance segmentation masks.21 Because the industrial deployment of segmentation models is often too slow for high-speed tracking, researchers routinely apply bounding box extraction algorithms to convert the fine-grained masks into rapid object detection coordinates.21 The immense value of the Kolektor dataset lies directly in its extreme difficulty; the defects are often highly subtle, and the overwhelming 89% normal image ratio forces convolutional models to develop highly sophisticated, robust feature extraction mechanisms to avoid false positive triggers.23 It is publicly accessible for free download through platforms such as Roboflow and GitHub, utilizing automated download scripts for immediate integration into Python pipelines.21

### **The Metal Surface Defect Dataset (MSDD) and Photometric Stereo**

Flat rolled steel and uniform commutators represent only a fraction of metallurgical production. Many metal blanks and castings present dynamic illumination challenges where extreme shadows, complex non-planar geometries, and varying surface reflectivity cause exceptionally high false detection rates in traditional 2D optical systems.3 The Metal Surface Defect Dataset (MSDD), available via GitHub and the Science Data Bank, represents a monumental breakthrough in addressing these specific optical complexities.3

#### **Photometric Stereo Acquisition**

To construct the massive MSDD repository, researchers employed a highly advanced Stroboscopic Illuminant Image Acquisition technique.3 Standard two-dimensional images natively lack depth information and surface normal directionality, making it incredibly challenging for a neural network to differentiate between a harmless cast shadow, an acceptable fluid stain, and a critical structural crack.28 The stroboscopic method utilizes a specially designed, multi-directional arrangement of illuminants and a sophisticated channel mixer to blend collected multi-channel images into composite RGB pseudo-color images.3

This photometric stereo approach effectively maps complex color space transformations directly to spatial domain transformations, neutralizing the confounding variables of metallic glare and deep geometric shadowing.3 This drastically enhances the three-dimensional resolution capability of the defect detection system.28 The dataset generated by this process is staggering in its scale, containing a total of 138,585 single-channel images alongside 9,239 mixed pseudo-color images.3 Directly addressing the industrial requirement for negative samples, the mixed pseudo-color dataset explicitly includes 5,746 completely defect-free normal images alongside 3,493 images containing physical defects.3

#### **Bounding Box Validation**

The MSDD covers eight highly specific defect types directly applicable to the automated visual inspection of casting-formed metal blanks and surfaces.3 Crucially, the annotation files are provided explicitly in the standard Pascal VOC format.28 This format provides the absolute bounding box coordinates defining the extremum corners of the damage, alongside the highly specific defect category.28

The utility and stability of the MSDD for bounding-box object detection research are comprehensively validated by its published performance benchmarks. Advanced universal single-stage and transformer-based object detection models, including FCOS, YOLOv5, YOLOv8, and the Real-Time DEtection TRansformer (RT-DETR), have been rigorously trained on this dataset.3 These architectures achieved an impressive mean Average Precision of 86.1%, confirming that the bounding box annotations and the high ratio of defect-free background images provide an optimal, mathematically stable training ground for state-of-the-art neural networks.3

## **Exhaustive Review of Foundational Research and Augmented Datasets**

While the aforementioned datasets natively contain the required distributions of normal images, the academic landscape is heavily saturated with several foundational, highly taxonomized datasets that natively lack normal images. However, due to their incredible depth of defect categorization and precise bounding boxes, they are considered mandatory benchmarks in the field. In modern deployment pipelines, these datasets are frequently merged with external repositories of normal images, or artificially augmented, to create robust, real-world training environments.14

### **The Northeastern University Surface Defect Database (NEU-DET)**

The Northeastern University (NEU) surface defect database is arguably the most widely cited, analyzed, and benchmarked repository in the entirety of steel defect detection literature.32 Published and maintained by the Surface Inspection Laboratory of Northeastern University, it focuses specifically on the complex optical characteristics of hot-rolled steel strips.33

#### **Defect Taxonomy and Optical Challenges**

The foundational NEU dataset contains exactly 1,800 grayscale images, maintained at a uniform, highly compressed resolution of 200 × 200 pixels.17 It categorizes metallurgical anomalies into six distinct types, ensuring perfect class balance with precisely 300 samples per class 17:

| NEU-DET Defect Class | Physical Manifestation |
| :---- | :---- |
| **Rolled-in Scale (RS)** | Hard oxide scale pressed deep into the metal surface during the rolling phase. |
| **Patches (Pa)** | Irregular, contiguous surface blemishes altering localized reflectivity. |
| **Crazing (Cr)** | A dense, interconnected network of fine, microscopic cracks on the surface. |
| **Pitted Surface (PS)** | Distinct, localized indentations, cavities, or craters in the metal matrix. |
| **Inclusion (In)** | Non-metallic foreign matter permanently bound into the surface of the metal. |
| **Scratches (Sc)** | Linear mechanical abrasions varying in depth and angular orientation. |

The NEU-DET dataset presents a highly specific, mathematically difficult challenge for convolutional networks: the simultaneous presence of extreme intra-class variation combined with profound inter-class similarity.17 For example, the physical appearance of a "scratch" can manifest horizontally along the rolling direction, vertically across the strip, or at an unpredictable slanted angle, drastically altering its pixel-level topology.17 Furthermore, due to the influence of ambient mill illumination and intrinsic material changes, the grayscale intensity of images within the exact same defect class can vary wildly.17 Conversely, distinct inter-class defects like rolled-in scale, crazing, and pitted surfaces share remarkably similar visual characteristics to the untrained algorithmic eye.17

The original iteration of this repository, named NEU-CLS, was dedicated strictly to whole-image classification and lacked any spatial localization data.34 Recognizing the industrial need for robotic localization, the subsequent NEU-DET iteration was released, providing meticulously annotated XML files containing precise bounding boxes, indicating both the absolute pixel location of the damage and the assigned class category.17 However, it must be explicitly noted that the core NEU-DET dataset is entirely comprised of defective images.35 To utilize it in a real-world scenario mimicking the Severstal distribution, researchers must artificially inject thousands of normal hot-rolled steel images, or utilize advanced transfer learning protocols where the background discrimination weights are learned from other repositories before fine-tuning on the NEU bounding boxes. It is freely available on Kaggle and GitHub.36

### **The GC10-DET Metallic Surface Defect Dataset**

Collected directly from a real-world industrial hot-rolled strip production line utilizing high-speed linear array CCD cameras (operating at extreme line speeds of up to 10 meters per second), the GC10-DET dataset expands significantly upon the taxonomy established by NEU-DET.6

#### **Extended Taxonomy and Pathological Failures**

The dataset consists of approximately 3,570 grayscale images meticulously annotated with 3,563 bounding box objects.38 It identifies ten highly specific defect classes, rooted deeply in mechanical engineering failure modes and operational mishaps 38:

| GC10-DET Defect Class | Physical Origin and Appearance |
| :---- | :---- |
| **Punching (Pu)** | Unwanted mechanical holes caused by equipment failure during specification punching. |
| **Weld Line (Wl)** | Distinct seams generated when two separate coils of a strip are welded together. |
| **Crescent Gap (Cg)** | Semi-circular edge defects resulting from faulty or misaligned cutting processes. |
| **Water Spot (Ws)** | Visual discoloration anomalies produced during the thermal drying phase. |
| **Oil Spot (Os)** | Mechanical lubricant contamination dripping onto the strip, affecting aesthetic quality. |
| **Waist Folding (Wf)** | Wrinkle-like folds across the strip caused by specific low-carbon metallurgical properties. |
| **Crease (Cr)** | Vertical or transverse folds across the strip caused directly by the uncoiling process. |
| **Rolled Pit (Rp)** | Periodic bulges or pits caused specifically by tension roll damage or work roll wear. |
| **Silk Spot (Ss) & Inclusion (In)** | Varied surface blemishes and pressed foreign matter. |

Like NEU-DET, the GC10-DET dataset provides exceptional bounding box annotations matched with whole-image specific classifications.39 However, it is also predominantly composed of defective images. Although 8 unlabeled, theoretically normal images are nominally included in the repository 39, this statistical percentage rounds down to 0% and is entirely insufficient for algorithmic background training. Consequently, it is frequently merged with other datasets by the open-source community. For example, the highly popular public "c5data2" repository hosted on Kaggle explicitly merges the core GC10-DET dataset with an influx of defect-free and varied online metal images to create custom, robust deployment pipelines containing normal samples alongside bounding boxes.14

### **The X-SDD Hot-Rolled Steel Strip Dataset**

The Xsteel Surface Defect Dataset (X-SDD) provides an alternative to the NEU benchmark, focusing on a slightly different subset of metallurgical failures observed in hot-rolling facilities.40 Containing 1,360 images with dimensions of 128 × 128 pixels in a 3-channel format, the dataset contains seven highly specific types of surface defects.41

Unlike the perfectly balanced 300-images-per-class NEU dataset, the X-SDD dataset explicitly embraces category imbalance, reflecting the reality that some defects occur far more frequently than others.41 The category with the largest amount of data is more than six times larger than the smallest.41 The specific defect categories include unique anomalies not found in NEU, such as **Iron Sheet Ash**, which is caused when accumulated metal dust, water, and oil fall onto the rolled parts from the mill equipment and become permanently embedded during subsequent rolling phases.40 Other categories include finishing roll printing, oxide scale of plate system, surface scratches, red iron sheet, and slag inclusions.40 While serving as a powerful supplement to NEU-CLS, the raw X-SDD is primarily a classification dataset and requires conversion or merging for rigorous bounding box localization.40

### **Auxiliary Real-World Formats: RSDDs and YSU\_CSC**

Expanding beyond standard rolled sheets, researchers have compiled specialized public datasets for other critical metal infrastructures:

1. **RSDDs (Rail Surface Defect Datasets):** Targeting the heavy transportation sector, RSDDs focuses on rail tracks.43 It is divided into Type I (captured from fast lanes, 67 images) and Type II (captured from normal/heavy transport tracks, 128 images).43 The dataset contains highly complex, noisy environmental backgrounds embedded with precise bounding box annotations.43 While the core defective dataset is relatively small, it is frequently paired with vast repositories of normal rail images for advanced few-shot learning and unsupervised anomaly detection algorithms.11  
2. **YSU\_CSC Copper Strip Dataset:** Highlighting multi-metal applicability, the YSU\_CSC dataset focuses on continuous copper strips, containing 2,400 images.45 Defects are classified into line marks, black spots, pits, edge cracks, holes, and peeling.45 Notably, the dataset methodology explicitly involves an initial discrimination step designed to distinguish between "perfect" (normal) and defective images, ensuring that the CNN models trained on it possess robust background classification capabilities before engaging in bounding box localization.7

### **The DAGM 2007 Synthetic Benchmark**

While purely synthetic, the DAGM 2007 dataset remains a foundational cornerstone in the literature for surface defect detection due to its rigorous, uncompromising structural approach to the normal-to-defective ratio.46 The dataset is categorized into ten distinct sub-datasets. Each sub-dataset is generated by a fundamentally different underlying texture model and a different defect model, simulating various industrial materials.10

To perfectly mimic the statistical rarity of defects in physical production lines, every individual sub-dataset consists of exactly 1,000 normal, non-defective background texture images juxtaposed against exactly 150 defective images.46 Each defective image features exactly one labeled anomaly.10 Originally provided with weak semantic labels in the form of ellipses roughly indicating the defective area, subsequent researchers have generated strict bounding box coordinates to adapt the dataset for modern deep learning detectors like the YOLO architecture.5 The values 0 and 255 denote the background and defective areas respectively in the associated 8-bit PNG mask files.10 When heavily modified YOLOv3 architectures are applied to this dataset, the network locates the regions with tight bounding rectangles, achieving superior computational efficiency without the need for sliding window proposals.5

## **Bounding Box Annotation Architectures and Algorithmic Integration**

The physical localization of a defect is executed through bounding boxes, which are simple geometric constraints that delineate the spatial extremities of the anomaly. The datasets reviewed in this report utilize various standardized annotation formats, dictating exactly how the bounding box coordinates are fed into the neural network's regression head.30 Understanding these formats is critical when downloading these datasets from Kaggle or GitHub to ensure compatibility with specific algorithms:

1. **Pascal VOC Format (Xmin, Ymin, Xmax, Ymax):** This format defines the absolute top-left and bottom-right corners of the bounding box based on absolute pixel coordinates.30 It is highly prevalent in legacy datasets like the MSDD and is straightforward for calculating the Intersection over Union (IoU) metric.28  
2. **YOLO Format (Xc, Yc, W, H):** Normalized to a scale between 0.0 and 1.0, this format defines the precise geometric center of the bounding box, followed by its width and height relative to the total image dimensions.30 Normalization is crucial for deep Convolutional Neural Networks as it inherently prevents the exploding gradient problem during training and allows the model to become scale-invariant across different image resolutions.47  
3. **COCO Format (Xmin, Ymin, W, H):** Similar to the Pascal VOC architecture but utilizes the top-left corner absolute coordinates alongside the absolute width and height of the box.30

### **Single-Stage vs. Two-Stage Detectors**

The architecture of these datasets—specifically the seamless integration of bounding boxes alongside negative background samples—directly informs the operational deep learning algorithms utilized in contemporary deployment. Object detection in high-speed industrial scenarios has largely coalesced around single-stage detectors due to their real-time inference capabilities.

Two-stage detectors, such as the widely known Faster R-CNN, operate by first generating a vast set of region proposals (areas mathematically likely to contain an object) and subsequently classifying these regions while refining the bounding box coordinates.48 While highly accurate at localizing defects, the massive computational overhead of generating thousands of region proposals renders them sub-optimal for high-speed rolling lines.48

Conversely, single-stage detectors like the YOLO (You Only Look Once) series divide the input image into a rigid grid system.15 Each grid cell simultaneously predicts bounding boxes and class probabilities.15 Anchor boxes—pre-defined spatial shapes based on the dataset's overarching statistics (often calculated via k-means clustering on the dataset's ground truth bounding boxes)—allow the model to adjust rapidly to drastically varying defect aspect ratios, such as differentiating a long, thin "scratch" from a perfectly round "pitted surface".15

### **The Unified Loss Function and Optimization**

Training a YOLO or DETR model on a highly complex dataset like Severstal or MSDD requires a unified loss function that simultaneously calculates and penalizes three separate mathematical components 15:

1. **Localization Loss:** Measures the exact coordinate error between the network's predicted bounding box and the human-annotated ground truth bounding box.  
2. **Confidence Loss:** Evaluates the model's certainty that a physical object actually exists within the generated box. This is precisely where normal, defect-free images exert their primary influence; the loss function aggressively penalizes the confidence score if a bounding box is drawn on a normal background.  
3. **Classification Loss:** Assesses whether the model correctly identified the specific defect class (e.g., distinguishing Crazing from an Inclusion) based on the image-level label.15

To optimize bounding box regression convergence without introducing extraneous, unstable loss terms, modern models applied to datasets like RSDDs often employ advanced geometric metrics such as Complete Intersection over Union (CIoU) or Inner-IoU, which utilize dynamically generated auxiliary bounding boxes to accelerate the training speed.50

## **Advanced Paradigms: The SteelDefectX Vision-Language Unified Framework**

A prevailing and critical issue in the datasets discussed thus far is academic fragmentation. A model trained to recognize a "rolled pit" on the GC10-DET dataset cannot natively transfer its localized knowledge to the "pitted surface" class in the NEU-DET dataset, despite the obvious physical and morphological similarities.17

To permanently resolve this dataset isolation and radically improve model interpretability, recent cutting-edge research has introduced **SteelDefectX**, a multi-form vision-language dataset that physically and semantically unifies four major public benchmarks: NEU, GC10, X-SDD, and S3D.31 Hosted publicly and freely on Hugging Face and GitHub, SteelDefectX comprises 7,778 images standardized to a uniform 256 × 256 resolution, spanning an incredible 25 harmonized steel surface defect categories.4

### **Multi-Level Textual and Spatial Annotations**

What fundamentally distinguishes the SteelDefectX framework from conventional bounding-box datasets is its deep integration of Natural Language Processing (NLP) to bridge low-level visual pixel observations with high-level semantic reasoning.31 Rather than merely outputting coordinates and a categorical integer, SteelDefectX trains multimodal models using four complementary text schemas alongside pixel-level masks 52:

* **T1 (Class-Level Description):** Defines category-level semantics, clearly articulating the defect name, representative visual attributes, and potential industrial causes (e.g., explicitly stating a defect is "caused by tension roll damage").31  
* **T2 (Natural Language Description):** Offers free-form, human-readable image-level descriptions detailing rich visual semantics unique to that specific image.52  
* **T3 (Structured Attributes):** Supplements simple bounding boxes with a highly structured nine-field attribute representation. It codifies Shape, Direction, Spatial Distribution, Number of Defects, absolute Position (e.g., explicitly coding "bottom-center"), Scale, Polarity, and visual Saliency.52  
* **T4 (Template Sentence):** Linearizes the structured T3 data into standardized, consistent text prompts explicitly designed for Large Language Model (LLM) processing.52

By augmenting spatial pixel-level masks with coarse-to-fine textual descriptions, SteelDefectX allows researchers to train massive multimodal architectures.52 These models can achieve zero-shot or few-shot generalization, learning the semantic, physical meaning of a "scratch" through text, and independently applying that spatial understanding to novel metallic domains without requiring exhaustive algorithmic re-training.53 This dataset represents the absolute frontier of open-source defect detection, fulfilling the user's requirement for exact localization, whole-image specific categorization, and real-world applicability in a single, unified package.

## **Conclusion and Strategic Dataset Deployment**

The landscape of open-source datasets for rolled metal surface defect detection is phenomenally rich, diverse, and rapidly evolving. The historic transition from simplistic, perfectly balanced classification datasets to massive, imbalanced, and meticulously localized object detection datasets signifies the true operational maturation of the field.

The requirement for both bounding box localization and specific image-level classification is mathematically non-negotiable for deploying multi-task single-stage detectors in real-world rolling mills. Furthermore, the inclusion of normal, defect-free images is the sole mechanism by which a neural network can learn to suppress false positives and construct an accurate confidence threshold against complex, reflective metallurgical backgrounds.

For practitioners utilizing platforms like Kaggle and GitHub, the deployment of these datasets should be strategically aligned with specific engineering goals. The **Severstal** dataset remains unparalleled for simulating the extreme class imbalance of actual flat-sheet production, offering over 11,000 normal images alongside convertible bounding box masks. For operations dealing with complex, non-planar geometries or extreme lighting variance, the **MSDD** provides an incredible 138,000-image repository utilizing photometric stereo to pierce through metallic glare. For foundational benchmarking of highly specific defect morphologies, the **NEU-DET** and **GC10-DET** datasets provide exceptional, rigorous taxonomies, provided the user artificially supplements them with normal background imagery to prevent algorithmic overfitting. Finally, for teams exploring the cutting edge of multimodal AI, the **SteelDefectX** unified framework provides the ultimate convergence of bounding box spatial awareness and deep, semantic metallurgical classification. Leveraging these comprehensive, public-domain repositories remains the primary and most effective vector through which automated computer vision models will achieve the speed, spatial precision, and cognitive depth required to fully automate global metallurgical quality assurance.

#### **Works cited**

1. Steel Surface Defect Recognition in Smart Manufacturing Using Deep Ensemble Transfer Learning-Based Techniques \- Mendeley, accessed on May 15, 2026, [https://www.mendeley.com/catalogue/01e4dd46-bbd4-3950-a6b5-8e9059db62df/](https://www.mendeley.com/catalogue/01e4dd46-bbd4-3950-a6b5-8e9059db62df/)  
2. Surface Defect Detection: Dataset & Papers \- GitHub, accessed on May 15, 2026, [https://github.com/SlideLucask/Surface-Defect-Detection-1](https://github.com/SlideLucask/Surface-Defect-Detection-1)  
3. Metal Surface Defect Dataset \- 科学数据银行, accessed on May 15, 2026, [https://www.scidb.cn/en/detail?dataSetId=3d739ddb4bdc439a9bf7ef550cae48d8](https://www.scidb.cn/en/detail?dataSetId=3d739ddb4bdc439a9bf7ef550cae48d8)  
4. SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection \- arXiv, accessed on May 15, 2026, [https://arxiv.org/html/2603.21824v1](https://arxiv.org/html/2603.21824v1)  
5. Research Article Surface Defect Detection with Modified Real-Time Detector YOLOv3 \- Macquarie University, accessed on May 15, 2026, [https://research-management.mq.edu.au/ws/portalfiles/portal/207279257/206980342.pdf](https://research-management.mq.edu.au/ws/portalfiles/portal/207279257/206980342.pdf)  
6. GC10-DET \- Kaggle, accessed on May 15, 2026, [https://www.kaggle.com/datasets/lirick/gc10-det](https://www.kaggle.com/datasets/lirick/gc10-det)  
7. Copper Strip Surface Defect Detection Model Based on Deep Convolutional Neural Network, accessed on May 15, 2026, [https://www.mdpi.com/2076-3417/11/19/8945](https://www.mdpi.com/2076-3417/11/19/8945)  
8. NEU Surface Defect Database examples | Download Scientific Diagram \- ResearchGate, accessed on May 15, 2026, [https://www.researchgate.net/figure/NEU-Surface-Defect-Database-examples\_fig1\_394035219](https://www.researchgate.net/figure/NEU-Surface-Defect-Database-examples_fig1_394035219)  
9. Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network \- MDPI, accessed on May 15, 2026, [https://www.mdpi.com/1424-8220/20/6/1562](https://www.mdpi.com/1424-8220/20/6/1562)  
10. DAGM 2007 competition dataset \- Kaggle, accessed on May 15, 2026, [https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)  
11. Few-shot Surface Defect Datasets \- 东北大学教师个人主页, accessed on May 15, 2026, [http://faculty.neu.edu.cn/songkechen/zh-CN/zhym/263269/list/index.htm](http://faculty.neu.edu.cn/songkechen/zh-CN/zhym/263269/list/index.htm)  
12. Severstal \- Dataset Ninja, accessed on May 15, 2026, [https://datasetninja.com/severstal](https://datasetninja.com/severstal)  
13. halmusaibeli/metal-defect-datasets \- GitHub, accessed on May 15, 2026, [https://github.com/halmusaibeli/metal-defect-datasets](https://github.com/halmusaibeli/metal-defect-datasets)  
14. Metal Surface Defects Dataset \- Kaggle, accessed on May 15, 2026, [https://www.kaggle.com/datasets/niclinn/c5data2](https://www.kaggle.com/datasets/niclinn/c5data2)  
15. YOLO-Based Defect Detection for Metal Sheets This work was supported in part by the Academia Sinica (AS) under Grant 235g Postdoctoral Scholar Program, in part by the National Science and Technology Council (NSTC) of Taiwan under Grant 113-2926-I-001-502-G. \- arXiv, accessed on May 15, 2026, [https://arxiv.org/html/2509.25659v1](https://arxiv.org/html/2509.25659v1)  
16. Detection of Scratch Defects on Metal Surfaces Based on MSDD-UNet \- MDPI, accessed on May 15, 2026, [https://www.mdpi.com/2079-9292/13/16/3241](https://www.mdpi.com/2079-9292/13/16/3241)  
17. 东北大学主页平台 Kechen Song--Home--NEU surface defect database, accessed on May 15, 2026, [http://faculty.neu.edu.cn/songkc/en/zdylm/263265](http://faculty.neu.edu.cn/songkc/en/zdylm/263265)  
18. Defect Detection for Metal Shaft Surfaces Based on an Improved YOLOv5 Algorithm and Transfer Learning \- PMC, accessed on May 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10098564/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10098564/)  
19. Severstal: Steel Defect Detection \- Kaggle, accessed on May 15, 2026, [https://www.kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection)  
20. Application of self-supervised learning in steel surface defect detection \- OAE Publishing Inc., accessed on May 15, 2026, [https://www.oaepublish.com/articles/jmi.2025.21](https://www.oaepublish.com/articles/jmi.2025.21)  
21. KolektorSDD2 \- Dataset Ninja, accessed on May 15, 2026, [https://datasetninja.com/kolektor-surface-defect-dataset-2](https://datasetninja.com/kolektor-surface-defect-dataset-2)  
22. CSDD: A Benchmark Dataset for Casting Surface Defect Detection and Segmentation, accessed on May 15, 2026, [https://www.ieee-jas.net/article/doi/10.1109/JAS.2025.125228](https://www.ieee-jas.net/article/doi/10.1109/JAS.2025.125228)  
23. kolektor-surface-defect-dataset-2/SUMMARY.md at main \- GitHub, accessed on May 15, 2026, [https://github.com/dataset-ninja/kolektor-surface-defect-dataset-2/blob/main/SUMMARY.md](https://github.com/dataset-ninja/kolektor-surface-defect-dataset-2/blob/main/SUMMARY.md)  
24. KolektorSDD2 Surface Defect Detection Dataset | Datasets | HyperAI, accessed on May 15, 2026, [https://hyper.ai/en/datasets/21545](https://hyper.ai/en/datasets/21545)  
25. Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2) \- ViCoS Lab, accessed on May 15, 2026, [https://www.vicos.si/resources/kolektorsdd2/](https://www.vicos.si/resources/kolektorsdd2/)  
26. Kolektor Data — Anomalib 2022 documentation, accessed on May 15, 2026, [https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/data/image/kolektor.html](https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/data/image/kolektor.html)  
27. dataset-ninja/kolektor-surface-defect-dataset-2 \- GitHub, accessed on May 15, 2026, [https://github.com/dataset-ninja/kolektor-surface-defect-dataset-2](https://github.com/dataset-ninja/kolektor-surface-defect-dataset-2)  
28. A dataset for surface defect detection on complex structured parts based on photometric stereo \- PMC, accessed on May 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11830798/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11830798/)  
29. (PDF) A dataset for surface defect detection on complex structured parts based on photometric stereo \- ResearchGate, accessed on May 15, 2026, [https://www.researchgate.net/publication/389046233\_A\_dataset\_for\_surface\_defect\_detection\_on\_complex\_structured\_parts\_based\_on\_photometric\_stereo](https://www.researchgate.net/publication/389046233_A_dataset_for_surface_defect_detection_on_complex_structured_parts_based_on_photometric_stereo)  
30. Bounding Box Annotation for Computer Vision Model Training | BasicAI's Blog, accessed on May 15, 2026, [https://www.basic.ai/blog-post/introduction-of-bounding-box-annotation](https://www.basic.ai/blog-post/introduction-of-bounding-box-annotation)  
31. SteelDefectX: A Multi-Form Vision-Language Dataset and Benchmark for Steel Surface Defect Analysis \- arXiv, accessed on May 15, 2026, [https://arxiv.org/html/2603.21824v2](https://arxiv.org/html/2603.21824v2)  
32. NEU-DET-Steel-Surface-Defect-Detection/Exploratory\_Data\_Analysis.ipynb at master, accessed on May 15, 2026, [https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/blob/master/Exploratory\_Data\_Analysis.ipynb](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/blob/master/Exploratory_Data_Analysis.ipynb)  
33. NEU-CLS \- figshare, accessed on May 15, 2026, [https://figshare.com/articles/dataset/NEU-CLS/28903550](https://figshare.com/articles/dataset/NEU-CLS/28903550)  
34. A Review of Metal Surface Defect Detection Technologies in Industrial Applications \- IEEE Xplore, accessed on May 15, 2026, [https://ieeexplore.ieee.org/iel8/6287639/10820123/10897983.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/10897983.pdf)  
35. Surface-Defect-Detection/README.md at master · Charmve/Surface ..., accessed on May 15, 2026, [https://github.com/Charmve/Surface-Defect-Detection/blob/master/README.md](https://github.com/Charmve/Surface-Defect-Detection/blob/master/README.md)  
36. NEU Surface Defect Database \- Kaggle, accessed on May 15, 2026, [https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)  
37. siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection: This project is about detecting defects on steel surface using Unet. The dataset used for this project is the NEU-DET database. \- GitHub, accessed on May 15, 2026, [https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection)  
38. Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network \- PMC, accessed on May 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7146379/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7146379/)  
39. GC10-DET \- Dataset Ninja, accessed on May 15, 2026, [https://datasetninja.com/gc10-det](https://datasetninja.com/gc10-det)  
40. X-SDD Database \- Kaggle, accessed on May 15, 2026, [https://www.kaggle.com/datasets/sayelabualigah/x-sdd](https://www.kaggle.com/datasets/sayelabualigah/x-sdd)  
41. X-SDD: A New Benchmark for Hot Rolled Steel Strip Surface Defects Detection \- MDPI, accessed on May 15, 2026, [https://www.mdpi.com/2073-8994/13/4/706](https://www.mdpi.com/2073-8994/13/4/706)  
42. X-SDD: A New Benchmark for Hot Rolled Steel Strip Surface Defects Detection, accessed on May 15, 2026, [https://www.researchgate.net/publication/350957745\_X-SDD\_A\_New\_Benchmark\_for\_Hot\_Rolled\_Steel\_Strip\_Surface\_Defects\_Detection](https://www.researchgate.net/publication/350957745_X-SDD_A_New_Benchmark_for_Hot_Rolled_Steel_Strip_Surface_Defects_Detection)  
43. GitHub \- Charmve/Surface-Defect-Detection: 目前最大的工业缺陷检测数据库及论文集 Constantly summarizing open source dataset and critical papers in the field of surface defect research which are of great importance., accessed on May 15, 2026, [https://github.com/Charmve/Surface-Defect-Detection](https://github.com/Charmve/Surface-Defect-Detection)  
44. FS-RSDD: Few-Shot Rail Surface Defect Detection with Prototype Learning \- PMC \- NIH, accessed on May 15, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10536558/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536558/)  
45. Surface defect dataset of copper strip (YSU\_CSC). \- ResearchGate, accessed on May 15, 2026, [https://www.researchgate.net/figure/Surface-defect-dataset-of-copper-strip-YSU-CSC\_fig2\_354863997](https://www.researchgate.net/figure/Surface-defect-dataset-of-copper-strip-YSU-CSC_fig2_354863997)  
46. Metal Surface Defect Detection Based on Few Defect Datasets \- AIP Publishing, accessed on May 15, 2026, [https://pubs.aip.org/aip/acp/article-pdf/doi/10.1063/1.5137871/13248962/020027\_1\_online.pdf](https://pubs.aip.org/aip/acp/article-pdf/doi/10.1063/1.5137871/13248962/020027_1_online.pdf)  
47. A Guide to Bounding Box Formats and How to Draw Them \- learnml.io, accessed on May 15, 2026, [https://www.learnml.io/posts/a-guide-to-bounding-box-formats/](https://www.learnml.io/posts/a-guide-to-bounding-box-formats/)  
48. Steel Surface Defect Detection Technology Based on YOLOv8-MGVS \- MDPI, accessed on May 15, 2026, [https://www.mdpi.com/2075-4701/15/2/109](https://www.mdpi.com/2075-4701/15/2/109)  
49. Ten types of defects in the DAGM 2007 dataset. \- ResearchGate, accessed on May 15, 2026, [https://www.researchgate.net/figure/Ten-types-of-defects-in-the-DAGM-2007-dataset\_fig11\_378774118](https://www.researchgate.net/figure/Ten-types-of-defects-in-the-DAGM-2007-dataset_fig11_378774118)  
50. Rail surface defect image from the RSDDs dataset. \- ResearchGate, accessed on May 15, 2026, [https://www.researchgate.net/figure/Rail-surface-defect-image-from-the-RSDDs-dataset\_fig2\_380119157](https://www.researchgate.net/figure/Rail-surface-defect-image-from-the-RSDDs-dataset_fig2_380119157)  
51. RSDNet: A New Multiscale Rail Surface Defect Detection Model \- MDPI, accessed on May 15, 2026, [https://www.mdpi.com/1424-8220/24/11/3579](https://www.mdpi.com/1424-8220/24/11/3579)  
52. GitHub \- Zhaosxian/SteelDefectX: A Multi-Form Vision-Language Dataset and Benchmark for Steel Surface Defect Analysis, accessed on May 15, 2026, [https://github.com/Zhaosxian/SteelDefectX](https://github.com/Zhaosxian/SteelDefectX)  
53. SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection \- ResearchGate, accessed on May 15, 2026, [https://www.researchgate.net/publication/403072986\_SteelDefectX\_A\_Coarse-to-Fine\_Vision-Language\_Dataset\_and\_Benchmark\_for\_Generalizable\_Steel\_Surface\_Defect\_Detection](https://www.researchgate.net/publication/403072986_SteelDefectX_A_Coarse-to-Fine_Vision-Language_Dataset_and_Benchmark_for_Generalizable_Steel_Surface_Defect_Detection)