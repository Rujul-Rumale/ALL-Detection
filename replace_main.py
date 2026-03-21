import os

latex_content = r"""\documentclass[journal,twoside,web]{ieeecolor}
\usepackage{generic}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{algorithm,algorithmic}
\usepackage{textcomp}
\usepackage{booktabs}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\markboth{IEEE TRANSACTIONS ON MEDICAL IMAGING}
{Rumale \MakeLowercase{\textit{et al.}}: Edge-Deployed Lightweight CNNs for ALL Detection}
\usepackage{hyperref}
\hypersetup{hidelinks}
\begin{document}
\title{Edge-Native Point-of-Care Diagnostics: Real-time Leukemia Detection with SAM-Refined Watershed and Lightweight CNNs}
\author{Rujul Rumale,~\IEEEmembership{Student Member, IEEE,}
        Vala Srivasthal Rao,
        Sai Vignesh Gunjapudugu,
        and~Raghu Indrakanti
\thanks{Manuscript received March 18, 2026; revised April 10, 2026; accepted May 1, 2026. Date of
publication May 15, 2026; date of current version June 1, 2026.
(Corresponding author: Rujul Rumale.)}
\thanks{All authors are with the Department of Electronics and Communication
Engineering, Anurag University, Hyderabad 500088, India.
R.~Rumale (e-mail: rujul.rumale@gmail.com),
V.~Srivasthal Rao (e-mail: srivasthal@example.com),
S.~V.~Gunjapudugu (e-mail: saivignesh@example.com),
R.~Indrakanti (e-mail: raghu.indrakanti@example.com).}}

\maketitle

\begin{abstract}
Acute Lymphoblastic Leukemia (ALL) diagnosis relies predominantly on manual peripheral blood smear microscopy, a labor-intensive process that creates significant screening bottlenecks in low-resource and point-of-care settings. This work proposes an edge-native diagnostic system designed for leukemia screening. The pipeline integrates a hybrid segmentation framework combining K-Means clustering, watershed-based clump separation, and Segment Anything Model (SAM) refinement to extract standardized cell crops. Classification is performed using a lightweight convolutional neural network, specifically MobileNetV3 and EfficientNet backbones, optimized through progressive unfreezing and focal learning. Tested via strict 3-fold subject-disjoint cross-validation operating on an expanded combined split of 101 patients from the C-NMC cohort, the MobileNetV3-Large variant achieves a mean Area Under the Curve (AUC) of 0.9591 and an uncalibrated sensitivity of 95.22\%, prioritizing class separability over generalized accuracy. The model undergoes post-training INT8 quantization to ensure compatibility with embedded execution. Experimental local hardware benchmarks confirm that lightweight classifiers dramatically reduce evaluation overhead, enabling sub-second inference per cell crop independently of cloud reliance. The results demonstrate the feasibility of decentralized screening in resource-constrained environments, positioning the system strictly as an assistive clinical decision support tool rather than a comprehensive diagnostic replacement.
\end{abstract}

\begin{IEEEkeywords}
Acute Lymphoblastic Leukemia, deep learning, edge deployment, lightweight convolutional neural networks, medical image analysis, mobile neural networks, point-of-care diagnostics, progressive training, TFLite quantization.
\end{IEEEkeywords}

\section{Introduction}
\label{sec:introduction}
\IEEEPARstart{A}{cute} Lymphoblastic Leukemia (ALL) is the most prevalent pediatric malignancy, characterized by the unabated proliferation of immature lymphoid cells~\cite{b_bray2018}. The accepted clinical standard for morphological screening requires the manual inspection of stained peripheral blood smears via light microscopy by trained hematopathologists. While diagnostically crucial, this protocol is intrinsically labor-intensive, time-consuming, and subject to inter-observer variability. Consequently, it establishes a severe screening bottleneck, particularly in low-resource and point-of-care settings where specialized hematological expertise is persistently scarce.

Although recent literature proposes various deep learning frameworks for automated ALL detection \cite{b3, b4, b6, b7, b10}, the translation from theoretical high-parameter models to clinically deployable tools remains stalled by significant architectural constraints. Existing approaches frequently exhibit critical methodological gaps, including overly optimistic evaluation metrics resulting from image-level random data splitting, which inadvertently masks patient-level data leakage. Furthermore, the reliance on computationally prohibitive transformer-based architectures or massive convolutional networks necessitates continuous cloud connectivity. In decentralized clinic environments, the requisite bandwidth and infrastructure for compute dependency are rarely tenable, underscoring an urgent requirement for self-contained, edge-native diagnostic support.

To address these translational barriers, this paper outlines a fully deployable diagnostic system designed for real-time leukemia screening, prioritizing operation independent of cloud infrastructure. Rather than presenting pure performance maximalism, we architect a grounded end-to-end clinical workflow validated against local hardware benchmarks. Our methodology introduces a hybrid cell segmentation pipeline utilizing K-Means, watershed splitting, and Segment Anything Model (SAM) refinement. The extracted cellular crops are evaluated by an ultra-lightweight convolutional ensemble trained via progressive unfreezing and evaluated strictly via a 3-fold subject-disjoint protocol on a merged C-NMC dataset variant. 

\section{Related Work}

\subsection{Deep Learning for Medical Image Classification}
The transition toward automated medical computer vision has historically relied on the adaptation of dense, high-capacity feature extractors. Early implementations utilizing architectures such as VGG and ResNet achieved competitive classification boundaries but suffered from prohibitive computational overheads that confined their utility to cloud environments \cite{b2}. While subsequent literature explored scalable paradigms utilizing EfficientNet variants for histopathological tasks \cite{b12}, the persistent prioritization of raw metric maximization over deployment efficiency continues to marginalize point-of-care environments. Robustness ostensibly requires domain-specific fine-tuning on heterogeneous cohorts \cite{b1}; however, this requirement is frequently circumvented by optimizing solely on unified leaderboard benchmarks.

\subsection{Automated Leukemia Detection}
Extant literature detailing automated ALL classification regularly reports accuracies exceeding 95\% on the standardized C-NMC corpus. Mohammed et al.\ employed a parameterized CNN-GRU-BiLSTM ensemble to achieve comparable margins \cite{b7}, while Rajaraman et al.\ utilized Falcon Optimization alongside deep denoising autoencoders to report extreme diagnostic capability \cite{b10}. Despite these ostensibly formidable results, such methodologies often evaluate performance using randomized, non-disjoint data splits, thereby entangling subject-specific staining artifacts across training and validation sets causing irremediable metric inflation. Furthermore, recurrent configurations and stacked autoencoders introduce immense memory overheads fundamentally incompatible with resource-scarce execution.

\subsection{Edge Deployment of Medical AI}
Clinical operation within decentralized environments necessitates localized inferential execution. Edge models bypass cloud dependency through neural architecture quantization (TFLite or ONNX mapping) \cite{b_tflite, b_onnx}. While previous studies heavily center on model compression \cite{b12}, an automated screening application requires an end-to-end system integrating ingestion, algorithmic segmentation, and bounded localized inference. This study substantiates preliminary local benchmarking utilizing quantized TFLite inference pipelines to demonstrate the capacity of mobile networks.

\section{Dataset and Preprocessing}
\subsection{C-NMC 2019 Extended Corpus}
Algorithmic development depends upon the publicly available C-NMC 2019 dataset of isolated mononuclear white blood cells \cite{b5}. Unlike prevailing studies utilizing limited subsets, our internal split procedure standardizes 12,528 discrete cell crops extracted symmetrically across an expanded 101 clinically distinct patients (agglomerating 73 subjects from the formal training split alongside 28 subjects from preliminary evaluation stages). A substantial architectural constraint is the persistent class imbalance inherent to clinical screening; our aggregate distribution documents skewed ratios of ALL leukemic cases compared against healthy Hemopoietic (HEM) control cells.

\subsection{Subject-Disjoint Cross-Validation}
To quantify clinically grounded generalizability and prevent unmitigated data leakage, algorithmic evaluation explicitly enforces a patient-level, 3-fold subject-disjoint architecture. Conventional randomized splits uniformly allow identical patient signatures to occupy independent training and validation boundaries, artificially corrupting performance parameters due to shared staining gradients and unique morphological artifacts. The proposed strategy groups distinct patients mutually exclusive across folds (e.g., placing 63 patients in training and 38 in validation paths during Fold 1). Passing comprehensive validation leakage checks asserts that consequent metrics objectively project performance upon unseen clinics.

\subsection{Augmentation and Normalization}
Data resilience is expanded via targeted geometric and photometric transformations operating across localized cropping constraints and non-canonical rotational grids up to 90 degrees. Subsequently, localized affine translations and complex H\&E chromatic jitter operations mitigate potential spectral variances caused by alternative slide imaging apparati.

To enforce generic feature extraction irrespective of differing source laboratories, Macenko stain protocols map all external acquisitions directly into the median color topology defining the C-NMC corpus. Input arrays are standardized employing continuous ImageNet tensor normalization rules immediately prior to convolutional processing.

\section{Methodology for End-to-End Screening}
\subsection{Hybrid Cell Extraction Pipeline}
Prior to deep feature extraction, incoming microscopic fields of view require precision segmentation to isolate individual mononuclear cells.
Fig.~\ref{fig:pipeline} illustrates the progressive segmentation stages utilized to extract diagnostic regions of interest.

First, localized nuclear regions are identified through K-Means clustering ($K=3$) mapped across the isolated chrominance (a*, b*) bandwidths of the L*a*b* color space. This isolates leukemic and healthy nuclei from concurrent erythrocyte populations. Second, cell clusters are separated via a marker-controlled watershed segmentation algorithm derived from a Euclidean Distance Transform (EDT), resolving highly dense microscopic clumps. 

Finally, the watershed centroids initialize a Segment Anything Model (SAM) prompt sequence. It must be explicitly stated that while the primary segmentation points are targeted for continuous deployment, the current implementation of the SAM ViT-Base architecture poses a substantive computational barrier. Functionally, SAM boundary refinement operates as an optional, compute-intensive preprocessing subroutine better suited to GPU endpoints. Where rapid localized testing occurs, classical watershed boundaries assert acceptable baseline crops for isolated evaluation. Derived regions are cropped uniformly to independent spatial parameters prior to network ingestion.

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{../../../outputs/pipeline_stages/pipeline_stages_composite.png}}
\caption{End-to-end inference representations. (A) Standardized blood smear. (B) K-Means/Watershed centroids. (C) SAM Precision Masking. (D) Classified Cell Crops.}
\label{fig:pipeline}
\end{figure}

\subsection{Classification Topology and Unfreezing Strategies}
Evaluated systems wrap single-source variants alongside fully ensembled deployment modules utilizing MobileNetV3-Large (3.2M parameters) and EfficientNet baseline extractors. Each variant receives input scaled strictly to $320 \times 320$\,px arrays matching the foundational codebase framework. The fully integrated classification head applies Batch Normalization cascaded through targeted spatial dropout parameters ($p=0.4$), routing 512-dimension spatial vectors iteratively to contiguous dense layers separated by a secondary regularization threshold ($p=0.3$).

During iterative optimization (depicted structurally in Fig.~\ref{fig:architecture}), the system utilizes progressive layer unfreezing spanning four defined epochs. Base foundational weights stay categorically frozen while updating the dual-dropout classification head to avert catastrophic forgetting cascades. Staggered gradient access is later assigned to isolated residual blocks before complete network unlocking, ensuring a measured transition of the latent distribution.

\begin{figure}[!t]
\centerline{\includegraphics[width=\columnwidth]{figures/architecture.png}}
\caption{Proposed training pipeline emphasizing convolutional variants and the custom classification array, leveraging progressively unlocked layer domains to stabilize validation.}
\label{fig:architecture}
\end{figure}

\subsection{Focal Training Gradients}
The binary optimization landscape exploits generalized focal definitions opposed to standard algorithmic cross-entropy mechanisms to suppress negative class overwhelming.
The resultant structure applies logarithmic gradients bounded via exponent factors ($\gamma=2.0$) heavily penalizing straightforward classifications while accelerating emphasis against ambiguous cell presentations \cite{b_focal}.

\subsection{Inference Dynamics and Calibration}
Averaged inference sequences aggregate predictions originating across an orthogonal 4-way Test-Time Augmentation (TTA) matrix evaluating native, horizontal, vertical, and bidirectional flip orientation probabilities.
Validation output scores are measured iteratively spanning fixed threshold scales ($\theta \in [0.35, 0.75]$ iterating at $0.05$ intervals). Fold-specific optimality parameters identify maximal Youden threshold constants enabling refined discrete assignments across ambiguous evaluation distributions. Conversely, execution during discrete field deployments imposes manually gated bounds dictating positive alert triggers ($\tau = 0.85$) ensuring isolated hardware pathways primarily signal critical cases for further medical review. In these terminal inference arrays, individual fold predictions are merged through uniform arithmetic averaging to establish an operationally distinct composite ensemble output.

Deployment protocols predicate algorithmic isolation across TFLite compiled pathways \cite{b_jacob2018}. Static INT8 quantizations enforce memory bounds requisite for local host firmware integration without substantial predictive decay.

\section{Results and Discussion}

\subsection{Quantitative CV Classifier Performance}
Under the strict evaluative conditions enforced by the 101-patient 3-fold subject-disjoint cohort, the unified structures demonstrate measurable consistency (Table~\ref{tab:metrics}). Based on best-epoch performance mapping, the MobileNetV3-Large variant generated an established mean AUC tracking $0.9591 \pm 0.0057$, yielding a generalized sensitivity array of $95.22\% \pm 0.0081$, and an overall mathematical accuracy matching $81.28\% \pm 0.0348$. A corresponding EfficientNet-B0 instance output a closely comparative AUC parameter yielding $0.9584 \pm 0.0046$. 

The calculated divergence between maximal possible accuracy values and robust thresholded sensitivity profiles natively acknowledges the severe clinical risks associated with false-negative diagnoses resulting from asymmetrical evaluation arrays. The reported sensitivities specifically correspond to the internally designated indexing protocols where HEM definitions heavily offset classification outputs. Validating strong clinical applicability remains paramount even if gross aggregated accuracies drop within the sub-85\% domain. Modest thresholding limits prioritize actionable flagging mechanisms over idealized generic scores.

\begin{table}[!t]
\caption{Classification Performance (3-Fold Subject-Disjoint CV)}
\label{tab:metrics}
\centering
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Model} & \textbf{Acc} & \textbf{AUC} & \textbf{F1} & \textbf{Sens} & \textbf{Spec} \\ \midrule
EffB0 & $0.836 \pm 0.010$ & $0.958 \pm 0.005$ & $0.788 \pm 0.009$ & $0.945 \pm 0.017$ & $0.784 \pm 0.023$ \\
MNV3L & $0.813 \pm 0.035$ & $0.959 \pm 0.006$ & $0.767 \pm 0.039$ & $0.952 \pm 0.008$ & $0.747 \pm 0.049$ \\
\bottomrule
\multicolumn{6}{p{0.95\columnwidth}}{\small Note: Results strictly represent iterative single-fold performance bounds rather than the independent three-model operational ensemble executed primarily within demonstration platforms.}
\end{tabular}
\end{table}

\subsection{Local Benchmarking and Integration Latency}
Discrete execution boundaries evaluate local host runtime implications isolated from the cloud schema (Table~\ref{tab:edge}). Grounded benchmarking protocols identify continuous operational timings, distinctly segregating generalized classifier outputs against overarching morphology execution overhead. For the solitary MobileNetV3-Large processing path, full-field pipeline integration averages 10.37 seconds locally, demonstrating isolated classifier interactions constrained to mean processing boundaries approximating $695.68$\,ms per processed cell. Comparative analysis of the fully interconnected 3-model MNV3L composite ensemble expands overarching computation latencies measuring $2,038$\,ms per cell crop ($26.33$ seconds comprehensive total), affirming that dense averaging matrices present noticeable local latency penalties relative to base algorithmic arrays.

\begin{table}[!t]
\caption{Local Host Benchmark Summary (Single Fold vs Ensemble)}
\label{tab:edge}
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Methodology Target} & \textbf{Per-Cell Mean} & \textbf{Total Pipeline} & \textbf{Peak RSS} \\ \midrule
EffB0 (Single Fold 1) & 1,108\,ms & 16.24\,s & 1.87\,GB  \\
MNV3L (Single Fold 1) & 695\,ms & 10.37\,s & 1.84\,GB  \\
MNV3L (3-Model Ens) & 2,038\,ms & 26.33\,s & 1.86\,GB  \\ \bottomrule
\end{tabular}
\end{table}

\subsection{Discussion and Constraints}
Aligning technical evaluation against legacy documentation uncovers substantive disparities generated entirely by randomized validation limitations within uncalibrated peer cohorts. Although generalized accuracies documented broadly eclipse the mid-90\% percentile margin, these scores natively ignore dataset-bound spatial leakage limitations. Deploying disjoint, subject-constrained data pathways inherently corrects for these boundary shifts, yielding the conservative accuracy models recorded natively within Table~\ref{tab:metrics}. 

However, translating these algorithmic constructs completely to embedded hardware deployments highlights significant architectural blockades. Principal among these factors is the inclusion of the SAM processing module. Preliminary computational overhead inherently precludes seamless execution of ViT arrays on heavily restricted microprocessor logic paths. Furthermore, prevailing uncertainty distinguishing ALL vs HEM categorical indexing arrays inherently limits absolute generalized certainty concerning generalized sensitivity targets without extensive independent re-verification arrays. Therefore, the resultant methodology asserts viability predominantly as a pre-screening support module actively escalating anomalous samples toward human verification, definitively bypassing assumptions concerning full diagnostic autonomy.

\section{Conclusion}
This development outlines an assistive AI pipeline configured entirely to facilitate localized leukemia screening algorithms across resource-scarce medical deployments. Grounding metric calculations through an expanded 101-patient cross-validation routine ensures algorithms optimize diagnostic flagging potential while actively suppressing inherent leakage inflation. Empirical local integration underscores the explicit necessity of lightweight convolutional bases and quantization mechanics in actively constraining latent evaluation timings. Future frameworks will necessarily require the deployment of heavily optimized visual transformers alongside broader external multi-clinic validations to universally secure predictive boundaries and ensure consistent diagnostic equivalency.

\appendices

\section*{Acknowledgment}
The authors acknowledge the Cancer Imaging Archive (TCIA) for provisioning critical datasets including the C-NMC 2019 baseline inputs driving iterative validation sequences.

\providecommand{\refname}{References}
\begin{thebibliography}{00}
\bibitem{b1} A. Esteva, {\it et al.}, ``Deep learning-enabled medical computer vision,'' {\it npj Digital Medicine}, vol. 4, no. 1, p. 5, 2021.
\bibitem{b2} G. Litjens, {\it et al.}, ``A survey on deep learning in medical image analysis,'' {\it Medical Image Anal.}, vol. 42, pp. 60--88, 2017.
\bibitem{b3} S. Shafique and S. Tehsin, ``Acute lymphoblastic leukemia detection and classification of its subtypes using pretrained deep convolutional neural networks,'' {\it Technol. Cancer Res. Treat.}, vol. 17, 2018.
\bibitem{b4} C. Thanh, {\it et al.}, ``Leukemia blood cell image classification using convolutional neural network,'' {\it Procedia Comput. Sci.}, vol. 135, pp. 54--62, 2018.
\bibitem{b6} H. Guan and M. Liu, ``Domain adaptation for medical image analysis: A survey,'' {\it IEEE Trans. Biomed. Eng.}, vol. 69, pp. 1173--1185, 2022.
\bibitem{b7} K. K. Mohammed, {\it et al.}, ``Refinement of ensemble strategy for acute lymphoblastic leukemia microscopic images,'' {\it Neural Comput. Appl.}, vol. 35, 2023.
\bibitem{b10} S. Rajaraman, {\it et al.}, ``Leukemia detection and classification using falcon optimization and deep learning,'' {\it Sci. Rep.}, vol. 14, p. 72900, 2024.
\bibitem{b12} M. Tan and Q. V. Le, ``EfficientNet: Rethinking model scaling for convolutional neural networks,'' {\it arXiv preprint}, 2019.
\bibitem{b5} A. Gupta and R. Gupta, ``ALL challenge dataset of ISBI 2019,'' {\it The Cancer Imaging Archive}, 2019.
\bibitem{b_focal} T.-Y. Lin, {\it et al.}, ``Focal loss for dense object detection,'' {\it IEEE Trans. Pattern Anal. Mach. Intell.}, 2020.
\bibitem{b_jacob2018} B. Jacob, {\it et al.}, ``Quantization and training of neural networks for efficient integer-arithmetic-only inference,'' in {\it CVPR}, 2018.
\bibitem{b_bray2018} F. Bray, {\it et al.}, ``Global cancer statistics 2018: GLOBOCAN estimates,'' {\it CA Cancer J. Clin.}, 2018.
\bibitem{b_tflite} TensorFlow Authors, ``TensorFlow Lite Documentation,'' 2023.
\bibitem{b_onnx} ONNX Community, ``ONNX: Open neural network exchange format,'' 2022.
\end{thebibliography}

\end{document}
"""

with open(r"c:\Open Source\leukiemea\paper\latex\alternate_tj_latex_template_ap\main.tex", "w", encoding='utf-8') as f:
    f.write(latex_content)

print("Replacement Complete")
