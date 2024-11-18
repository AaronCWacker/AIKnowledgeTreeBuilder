# Query: Current CRISPR RNA editing 🧪

## AI Response
### 🔎 Current CRISPR RNA editing 🧪

 technology is limited to editing DNA sequences that are 1-2 base pairs apart. This is because the Cas9 enzyme used in CRISPR-Cas9 systems is too large to fit between two DNA sequences that are farther apart.

To overcome this limitation, researchers have developed a new CRISPR RNA editing technology called CRISPR-Cas12a, which uses a smaller Cas12a enzyme that can fit between two DNA sequences that are farther apart.

CRISPR-Cas12a has been shown to be effective in editing DNA sequences that are up to 20 base pairs apart, which is a significant improvement over the current CRISPR-Cas9 technology. This new technology has the potential to expand the range of genetic diseases that can be treated using CRISPR RNA editing technology.

In a recent study, researchers used CRISPR-Cas12a to edit a genetic mutation that causes Duchenne muscular dystrophy (DMD), a severe muscle-wasting disease. The mutation is located in a gene called dystrophin, which is responsible for producing a protein that is essential for muscle function.

Using CRISPR-Cas12a, the researchers were able to correct the mutation in patient-derived cells, restoring the production of the dystrophin protein. The corrected cells were then transplanted into mice, where they were able to improve muscle function.

The study suggests that CRISPR-Cas12a has the potential to be used as a treatment for DMD and other genetic diseases that are caused by mutations in large genes. However, more research is needed to determine the safety and efficacy of this new technology in humans.

Overall, the development of CRISPR-Cas12a represents a significant advancement in CRISPR RNA editing technology, expanding the range of genetic diseases that can be treated using this approach. While more research is needed, this new technology holds promise for the treatment of a wide range of genetic diseases.</s>
# 🩺🔍 Search Results
### 05 Mar 2024 | [CRISPR: Ensemble Model](https://arxiv.org/abs/2403.03018) | [⬇️](https://arxiv.org/pdf/2403.03018)
*Mohammad Rostami, Amin Ghariyazi, Hamed Dashti, Mohammad Hossein  Rohban, Hamid R. Rabiee* 

  Clustered Regularly Interspaced Short Palindromic Repeats (CRISPR) is a gene
editing technology that has revolutionized the fields of biology and medicine.
However, one of the challenges of using CRISPR is predicting the on-target
efficacy and off-target sensitivity of single-guide RNAs (sgRNAs). This is
because most existing methods are trained on separate datasets with different
genes and cells, which limits their generalizability. In this paper, we propose
a novel ensemble learning method for sgRNA design that is accurate and
generalizable. Our method combines the predictions of multiple machine learning
models to produce a single, more robust prediction. This approach allows us to
learn from a wider range of data, which improves the generalizability of our
model. We evaluated our method on a benchmark dataset of sgRNA designs and
found that it outperformed existing methods in terms of both accuracy and
generalizability. Our results suggest that our method can be used to design
sgRNAs with high sensitivity and specificity, even for new genes or cells. This
could have important implications for the clinical use of CRISPR, as it would
allow researchers to design more effective and safer treatments for a variety
of diseases.

---------------

### 10 Jul 2023 | [Model-Driven Engineering for Artificial Intelligence -- A Systematic  Literature Review](https://arxiv.org/abs/2307.04599) | [⬇️](https://arxiv.org/pdf/2307.04599)
*Simon Raedler, Luca Berardinelli, Karolin Winter, Abbas Rahimi,  Stefanie Rinderle-Ma* 

  Objective: This study aims to investigate the existing body of knowledge in
the field of Model-Driven Engineering MDE in support of AI (MDE4AI) to sharpen
future research further and define the current state of the art.
  Method: We conducted a Systemic Literature Review (SLR), collecting papers
from five major databases resulting in 703 candidate studies, eventually
retaining 15 primary studies. Each primary study will be evaluated and
discussed with respect to the adoption of (1) MDE principles and practices and
(2) the phases of AI development support aligned with the stages of the
CRISP-DM methodology.
  Results: The study's findings show that the pillar concepts of MDE
(metamodel, concrete syntax and model transformation), are leveraged to define
domain-specific languages (DSL) explicitly addressing AI concerns. Different
MDE technologies are used, leveraging different language workbenches. The most
prominent AI-related concerns are training and modeling of the AI algorithm,
while minor emphasis is given to the time-consuming preparation of the data
sets. Early project phases that support interdisciplinary communication of
requirements, such as the CRISP-DM \textit{Business Understanding} phase, are
rarely reflected.
  Conclusion: The study found that the use of MDE for AI is still in its early
stages, and there is no single tool or method that is widely used.
Additionally, current approaches tend to focus on specific stages of
development rather than providing support for the entire development process.
As a result, the study suggests several research directions to further improve
the use of MDE for AI and to guide future research in this area.

---------------

### 20 Feb 2024 | [dotears: Scalable, consistent DAG estimation using observational and  interventional data](https://arxiv.org/abs/2305.19215) | [⬇️](https://arxiv.org/pdf/2305.19215)
*Albert Xue, Jingyou Rao, Sriram Sankararaman, Harold Pimentel* 

  New biological assays like Perturb-seq link highly parallel CRISPR
interventions to a high-dimensional transcriptomic readout, providing insight
into gene regulatory networks. Causal gene regulatory networks can be
represented by directed acyclic graph (DAGs), but learning DAGs from
observational data is complicated by lack of identifiability and a
combinatorial solution space. Score-based structure learning improves practical
scalability of inferring DAGs. Previous score-based methods are sensitive to
error variance structure; on the other hand, estimation of error variance is
difficult without prior knowledge of structure. Accordingly, we present
$\texttt{dotears}$ [doo-tairs], a continuous optimization framework which
leverages observational and interventional data to infer a single causal
structure, assuming a linear Structural Equation Model (SEM).
$\texttt{dotears}$ exploits structural consequences of hard interventions to
give a marginal estimate of exogenous error structure, bypassing the circular
estimation problem. We show that $\texttt{dotears}$ is a provably consistent
estimator of the true DAG under mild assumptions. $\texttt{dotears}$
outperforms other methods in varied simulations, and in real data infers edges
that validate with higher precision and recall than state-of-the-art methods
through differential expression tests and high-confidence protein-protein
interactions.

---------------

### 29 Sep 2023 | [Interpretable neural architecture search and transfer learning for  understanding CRISPR/Cas9 off-target enzymatic reactions](https://arxiv.org/abs/2305.11917) | [⬇️](https://arxiv.org/pdf/2305.11917)
*Zijun Zhang, Adam R. Lamson, Michael Shelley, Olga Troyanskaya* 

  Finely-tuned enzymatic pathways control cellular processes, and their
dysregulation can lead to disease. Creating predictive and interpretable models
for these pathways is challenging because of the complexity of the pathways and
of the cellular and genomic contexts. Here we introduce Elektrum, a deep
learning framework which addresses these challenges with data-driven and
biophysically interpretable models for determining the kinetics of biochemical
systems. First, it uses in vitro kinetic assays to rapidly hypothesize an
ensemble of high-quality Kinetically Interpretable Neural Networks (KINNs) that
predict reaction rates. It then employs a novel transfer learning step, where
the KINNs are inserted as intermediary layers into deeper convolutional neural
networks, fine-tuning the predictions for reaction-dependent in vivo outcomes.
Elektrum makes effective use of the limited, but clean in vitro data and the
complex, yet plentiful in vivo data that captures cellular context. We apply
Elektrum to predict CRISPR-Cas9 off-target editing probabilities and
demonstrate that Elektrum achieves state-of-the-art performance, regularizes
neural network architectures, and maintains physical interpretability.

---------------

### 11 Dec 2023 | [SmartEdit: Exploring Complex Instruction-based Image Editing with  Multimodal Large Language Models](https://arxiv.org/abs/2312.06739) | [⬇️](https://arxiv.org/pdf/2312.06739)
*Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun,  Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, Ying Shan* 

  Current instruction-based editing methods, such as InstructPix2Pix, often
fail to produce satisfactory results in complex scenarios due to their
dependence on the simple CLIP text encoder in diffusion models. To rectify
this, this paper introduces SmartEdit, a novel approach to instruction-based
image editing that leverages Multimodal Large Language Models (MLLMs) to
enhance their understanding and reasoning capabilities. However, direct
integration of these elements still faces challenges in situations requiring
complex reasoning. To mitigate this, we propose a Bidirectional Interaction
Module that enables comprehensive bidirectional information interactions
between the input image and the MLLM output. During training, we initially
incorporate perception data to boost the perception and understanding
capabilities of diffusion models. Subsequently, we demonstrate that a small
amount of complex instruction editing data can effectively stimulate
SmartEdit's editing capabilities for more complex instructions. We further
construct a new evaluation dataset, Reason-Edit, specifically tailored for
complex instruction-based image editing. Both quantitative and qualitative
results on this evaluation dataset indicate that our SmartEdit surpasses
previous methods, paving the way for the practical application of complex
instruction-based image editing.

---------------

### 24 Nov 2023 | [CRISP: Hybrid Structured Sparsity for Class-aware Model Pruning](https://arxiv.org/abs/2311.14272) | [⬇️](https://arxiv.org/pdf/2311.14272)
*Shivam Aggarwal, Kuluhan Binici, Tulika Mitra* 

  Machine learning pipelines for classification tasks often train a universal
model to achieve accuracy across a broad range of classes. However, a typical
user encounters only a limited selection of classes regularly. This disparity
provides an opportunity to enhance computational efficiency by tailoring models
to focus on user-specific classes. Existing works rely on unstructured pruning,
which introduces randomly distributed non-zero values in the model, making it
unsuitable for hardware acceleration. Alternatively, some approaches employ
structured pruning, such as channel pruning, but these tend to provide only
minimal compression and may lead to reduced model accuracy. In this work, we
propose CRISP, a novel pruning framework leveraging a hybrid structured
sparsity pattern that combines both fine-grained N:M structured sparsity and
coarse-grained block sparsity. Our pruning strategy is guided by a
gradient-based class-aware saliency score, allowing us to retain weights
crucial for user-specific classes. CRISP achieves high accuracy with minimal
memory consumption for popular models like ResNet-50, VGG-16, and MobileNetV2
on ImageNet and CIFAR-100 datasets. Moreover, CRISP delivers up to 14$\times$
reduction in latency and energy consumption compared to existing pruning
methods while maintaining comparable accuracy. Our code is available at
https://github.com/shivmgg/CRISP/.

---------------

### 29 Jan 2019 | [Revised JNLPBA Corpus: A Revised Version of Biomedical NER Corpus for  Relation Extraction Task](https://arxiv.org/abs/1901.10219) | [⬇️](https://arxiv.org/pdf/1901.10219)
*Ming-Siang Huang, Po-Ting Lai, Richard Tzong-Han Tsai, Wen-Lian Hsu* 

  The advancement of biomedical named entity recognition (BNER) and biomedical
relation extraction (BRE) researches promotes the development of text mining in
biological domains. As a cornerstone of BRE, robust BNER system is required to
identify the mentioned NEs in plain texts for further relation extraction
stage. However, the current BNER corpora, which play important roles in these
tasks, paid less attention to achieve the criteria for BRE task. In this study,
we present Revised JNLPBA corpus, the revision of JNLPBA corpus, to broaden the
applicability of a NER corpus from BNER to BRE task. We preserve the original
entity types including protein, DNA, RNA, cell line and cell type while all the
abstracts in JNLPBA corpus are manually curated by domain experts again basis
on the new annotation guideline focusing on the specific NEs instead of general
terms. Simultaneously, several imperfection issues in JNLPBA are pointed out
and made up in the new corpus. To compare the adaptability of different NER
systems in Revised JNLPBA and JNLPBA corpora, the F1-measure was measured in
three open sources NER systems including BANNER, Gimli and NERSuite. In the
same circumstance, all the systems perform average 10% better in Revised JNLPBA
than in JNLPBA. Moreover, the cross-validation test is carried out which we
train the NER systems on JNLPBA/Revised JNLPBA corpora and access the
performance in both protein-protein interaction extraction (PPIE) and
biomedical event extraction (BEE) corpora to confirm that the newly refined
Revised JNLPBA is a competent NER corpus in biomedical relation application.
The revised JNLPBA corpus is freely available at
iasl-btm.iis.sinica.edu.tw/BNER/Content/Revised_JNLPBA.zip.

---------------

### 28 Nov 2011 | [A kernel-based framework for learning graded relations from data](https://arxiv.org/abs/1111.6473) | [⬇️](https://arxiv.org/pdf/1111.6473)
*Willem Waegeman, Tapio Pahikkala, Antti Airola, Tapio Salakoski,  Michiel Stock, Bernard De Baets* 

  Driven by a large number of potential applications in areas like
bioinformatics, information retrieval and social network analysis, the problem
setting of inferring relations between pairs of data objects has recently been
investigated quite intensively in the machine learning community. To this end,
current approaches typically consider datasets containing crisp relations, so
that standard classification methods can be adopted. However, relations between
objects like similarities and preferences are often expressed in a graded
manner in real-world applications. A general kernel-based framework for
learning relations from data is introduced here. It extends existing approaches
because both crisp and graded relations are considered, and it unifies existing
approaches because different types of graded relations can be modeled,
including symmetric and reciprocal relations. This framework establishes
important links between recent developments in fuzzy set theory and machine
learning. Its usefulness is demonstrated through various experiments on
synthetic and real-world data.

---------------

### 26 Feb 2019 | [Extending Recurrent Neural Aligner for Streaming End-to-End Speech  Recognition in Mandarin](https://arxiv.org/abs/1806.06342) | [⬇️](https://arxiv.org/pdf/1806.06342)
*Linhao Dong, Shiyu Zhou, Wei Chen, Bo Xu* 

  End-to-end models have been showing superiority in Automatic Speech
Recognition (ASR). At the same time, the capacity of streaming recognition has
become a growing requirement for end-to-end models. Following these trends, an
encoder-decoder recurrent neural network called Recurrent Neural Aligner (RNA)
has been freshly proposed and shown its competitiveness on two English ASR
tasks. However, it is not clear if RNA can be further improved and applied to
other spoken language. In this work, we explore the applicability of RNA in
Mandarin Chinese and present four effective extensions: In the encoder, we
redesign the temporal down-sampling and introduce a powerful convolutional
structure. In the decoder, we utilize a regularizer to smooth the output
distribution and conduct joint training with a language model. On two Mandarin
Chinese conversational telephone speech recognition (MTS) datasets, our
Extended-RNA obtains promising performance. Particularly, it achieves 27.7%
character error rate (CER), which is superior to current state-of-the-art
result on the popular HKUST task.

---------------

### 16 Nov 2023 | [CRISPR: Eliminating Bias Neurons from an Instruction-following Language  Model](https://arxiv.org/abs/2311.09627) | [⬇️](https://arxiv.org/pdf/2311.09627)
*Nakyeong Yang, Taegwan Kang and Kyomin Jung* 

  Large language models (LLMs) executing tasks through instruction-based
prompts often face challenges stemming from distribution differences between
user instructions and training instructions. This leads to distractions and
biases, especially when dealing with inconsistent dynamic labels. In this
paper, we introduces a novel bias mitigation method, CRISPR, designed to
alleviate instruction-label biases in LLMs. CRISPR utilizes attribution methods
to identify bias neurons influencing biased outputs and employs pruning to
eliminate the bias neurons. Experimental results demonstrate the method's
effectiveness in mitigating biases in instruction-based prompting, enhancing
language model performance on social bias benchmarks without compromising
pre-existing knowledge. CRISPR proves highly practical, model-agnostic,
offering flexibility in adapting to evolving social biases.

---------------

### 15 Jan 2014 | [Bounds Arc Consistency for Weighted CSPs](https://arxiv.org/abs/1401.3481) | [⬇️](https://arxiv.org/pdf/1401.3481)
*Matthias Zytnicki, Christine Gaspin, Simon de Givry, Thomas Schiex* 

  The Weighted Constraint Satisfaction Problem (WCSP) framework allows
representing and solving problems involving both hard constraints and cost
functions. It has been applied to various problems, including resource
allocation, bioinformatics, scheduling, etc. To solve such problems, solvers
usually rely on branch-and-bound algorithms equipped with local consistency
filtering, mostly soft arc consistency. However, these techniques are not well
suited to solve problems with very large domains. Motivated by the resolution
of an RNA gene localization problem inside large genomic sequences, and in the
spirit of bounds consistency for large domains in crisp CSPs, we introduce soft
bounds arc consistency, a new weighted local consistency specifically designed
for WCSP with very large domains. Compared to soft arc consistency, BAC
provides significantly improved time and space asymptotic complexity. In this
paper, we show how the semantics of cost functions can be exploited to further
improve the time complexity of BAC. We also compare both in theory and in
practice the efficiency of BAC on a WCSP with bounds consistency enforced on a
crisp CSP using cost variables. On two different real problems modeled as WCSP,
including our RNA gene localization problem, we observe that maintaining bounds
arc consistency outperforms arc consistency and also improves over bounds
consistency enforced on a constraint model with cost variables.

---------------

### 04 Dec 2023 | [ChemSpaceAL: An Efficient Active Learning Methodology Applied to  Protein-Specific Molecular Generation](https://arxiv.org/abs/2309.05853) | [⬇️](https://arxiv.org/pdf/2309.05853)
*Gregory W. Kyro, Anton Morgunov, Rafael I. Brent, Victor S. Batista* 

  The incredible capabilities of generative artificial intelligence models have
inevitably led to their application in the domain of drug discovery. Within
this domain, the vastness of chemical space motivates the development of more
efficient methods for identifying regions with molecules that exhibit desired
characteristics. In this work, we present a computationally efficient active
learning methodology that requires evaluation of only a subset of the generated
data in the constructed sample space to successfully align a generative model
with respect to a specified objective. We demonstrate the applicability of this
methodology to targeted molecular generation by fine-tuning a GPT-based
molecular generator toward a protein with FDA-approved small-molecule
inhibitors, c-Abl kinase. Remarkably, the model learns to generate molecules
similar to the inhibitors without prior knowledge of their existence, and even
reproduces two of them exactly. We also show that the methodology is effective
for a protein without any commercially available small-molecule inhibitors, the
HNH domain of the CRISPR-associated protein 9 (Cas9) enzyme. We believe that
the inherent generality of this method ensures that it will remain applicable
as the exciting field of in silico molecular generation evolves. To facilitate
implementation and reproducibility, we have made all of our software available
through the open-source ChemSpaceAL Python package.

---------------

### 22 Sep 2023 | [FluentEditor: Text-based Speech Editing by Considering Acoustic and  Prosody Consistency](https://arxiv.org/abs/2309.11725) | [⬇️](https://arxiv.org/pdf/2309.11725)
*Rui Liu, Jiatian Xi, Ziyue Jiang and Haizhou Li* 

  Text-based speech editing (TSE) techniques are designed to enable users to
edit the output audio by modifying the input text transcript instead of the
audio itself. Despite much progress in neural network-based TSE techniques, the
current techniques have focused on reducing the difference between the
generated speech segment and the reference target in the editing region,
ignoring its local and global fluency in the context and original utterance. To
maintain the speech fluency, we propose a fluency speech editing model, termed
\textit{FluentEditor}, by considering fluency-aware training criterion in the
TSE training. Specifically, the \textit{acoustic consistency constraint} aims
to smooth the transition between the edited region and its neighboring acoustic
segments consistent with the ground truth, while the \textit{prosody
consistency constraint} seeks to ensure that the prosody attributes within the
edited regions remain consistent with the overall style of the original
utterance. The subjective and objective experimental results on VCTK
demonstrate that our \textit{FluentEditor} outperforms all advanced baselines
in terms of naturalness and fluency. The audio samples and code are available
at \url{https://github.com/Ai-S2-Lab/FluentEditor}.

---------------

### 11 Oct 2016 | [Deep Learning Assessment of Tumor Proliferation in Breast Cancer  Histological Images](https://arxiv.org/abs/1610.03467) | [⬇️](https://arxiv.org/pdf/1610.03467)
*Manan Shah, Christopher Rubadue, David Suster, Dayong Wang* 

  Current analysis of tumor proliferation, the most salient prognostic
biomarker for invasive breast cancer, is limited to subjective mitosis counting
by pathologists in localized regions of tissue images. This study presents the
first data-driven integrative approach to characterize the severity of tumor
growth and spread on a categorical and molecular level, utilizing multiple
biologically salient deep learning classifiers to develop a comprehensive
prognostic model. Our approach achieves pathologist-level performance on
three-class categorical tumor severity prediction. It additionally pioneers
prediction of molecular expression data from a tissue image, obtaining a
Spearman's rank correlation coefficient of 0.60 with ex vivo mean calculated
RNA expression. Furthermore, our framework is applied to identify over two
hundred unprecedented biomarkers critical to the accurate assessment of tumor
proliferation, validating our proposed integrative pipeline as the first to
holistically and objectively analyze histopathological images.

---------------

### 22 Oct 2021 | [Mechanistic Interpretation of Machine Learning Inference: A Fuzzy  Feature Importance Fusion Approach](https://arxiv.org/abs/2110.11713) | [⬇️](https://arxiv.org/pdf/2110.11713)
*Divish Rengasamy, Jimiama M. Mase, Mercedes Torres Torres, Benjamin  Rothwell, David A. Winkler, Grazziela P. Figueredo* 

  With the widespread use of machine learning to support decision-making, it is
increasingly important to verify and understand the reasons why a particular
output is produced. Although post-training feature importance approaches assist
this interpretation, there is an overall lack of consensus regarding how
feature importance should be quantified, making explanations of model
predictions unreliable. In addition, many of these explanations depend on the
specific machine learning approach employed and on the subset of data used when
calculating feature importance. A possible solution to improve the reliability
of explanations is to combine results from multiple feature importance
quantifiers from different machine learning approaches coupled with
re-sampling. Current state-of-the-art ensemble feature importance fusion uses
crisp techniques to fuse results from different approaches. There is, however,
significant loss of information as these approaches are not context-aware and
reduce several quantifiers to a single crisp output. More importantly, their
representation of 'importance' as coefficients is misleading and
incomprehensible to end-users and decision makers. Here we show how the use of
fuzzy data fusion methods can overcome some of the important limitations of
crisp fusion methods.

---------------

### 12 Jul 2023 | [Improving Chest X-Ray Report Generation by Leveraging Warm Starting](https://arxiv.org/abs/2201.09405) | [⬇️](https://arxiv.org/pdf/2201.09405)
*Aaron Nicolson, Jason Dowling, and Bevan Koopman* 

  Automatically generating a report from a patient's Chest X-Rays (CXRs) is a
promising solution to reducing clinical workload and improving patient care.
However, current CXR report generators -- which are predominantly
encoder-to-decoder models -- lack the diagnostic accuracy to be deployed in a
clinical setting. To improve CXR report generation, we investigate warm
starting the encoder and decoder with recent open-source computer vision and
natural language processing checkpoints, such as the Vision Transformer (ViT)
and PubMedBERT. To this end, each checkpoint is evaluated on the MIMIC-CXR and
IU X-Ray datasets. Our experimental investigation demonstrates that the
Convolutional vision Transformer (CvT) ImageNet-21K and the Distilled
Generative Pre-trained Transformer 2 (DistilGPT2) checkpoints are best for warm
starting the encoder and decoder, respectively. Compared to the
state-of-the-art ($\mathcal{M}^2$ Transformer Progressive), CvT2DistilGPT2
attained an improvement of 8.3\% for CE F-1, 1.8\% for BLEU-4, 1.6\% for
ROUGE-L, and 1.0\% for METEOR. The reports generated by CvT2DistilGPT2 have a
higher similarity to radiologist reports than previous approaches. This
indicates that leveraging warm starting improves CXR report generation. Code
and checkpoints for CvT2DistilGPT2 are available at
https://github.com/aehrc/cvt2distilgpt2.

---------------

### 14 Jun 2023 | [Improving Code-Switching and Named Entity Recognition in ASR with Speech  Editing based Data Augmentation](https://arxiv.org/abs/2306.08588) | [⬇️](https://arxiv.org/pdf/2306.08588)
*Zheng Liang, Zheshu Song, Ziyang Ma, Chenpeng Du, Kai Yu, Xie Chen* 

  Recently, end-to-end (E2E) automatic speech recognition (ASR) models have
made great strides and exhibit excellent performance in general speech
recognition. However, there remain several challenging scenarios that E2E
models are not competent in, such as code-switching and named entity
recognition (NER). Data augmentation is a common and effective practice for
these two scenarios. However, the current data augmentation methods mainly rely
on audio splicing and text-to-speech (TTS) models, which might result in
discontinuous, unrealistic, and less diversified speech. To mitigate these
potential issues, we propose a novel data augmentation method by applying the
text-based speech editing model. The augmented speech from speech editing
systems is more coherent and diversified, also more akin to real speech. The
experimental results on code-switching and NER tasks show that our proposed
method can significantly outperform the audio splicing and neural TTS based
data augmentation systems.

---------------

### 22 Oct 2021 | [GeneDisco: A Benchmark for Experimental Design in Drug Discovery](https://arxiv.org/abs/2110.11875) | [⬇️](https://arxiv.org/pdf/2110.11875)
*Arash Mehrjou, Ashkan Soleymani, Andrew Jesson, Pascal Notin, Yarin  Gal, Stefan Bauer, Patrick Schwab* 

  In vitro cellular experimentation with genetic interventions, using for
example CRISPR technologies, is an essential step in early-stage drug discovery
and target validation that serves to assess initial hypotheses about causal
associations between biological mechanisms and disease pathologies. With
billions of potential hypotheses to test, the experimental design space for in
vitro genetic experiments is extremely vast, and the available experimental
capacity - even at the largest research institutions in the world - pales in
relation to the size of this biological hypothesis space. Machine learning
methods, such as active and reinforcement learning, could aid in optimally
exploring the vast biological space by integrating prior knowledge from various
information sources as well as extrapolating to yet unexplored areas of the
experimental design space based on available data. However, there exist no
standardised benchmarks and data sets for this challenging task and little
research has been conducted in this area to date. Here, we introduce GeneDisco,
a benchmark suite for evaluating active learning algorithms for experimental
design in drug discovery. GeneDisco contains a curated set of multiple publicly
available experimental data sets as well as open-source implementations of
state-of-the-art active learning policies for experimental design and
exploration.

---------------

### 27 Jan 2024 | [Dual-Perspective Semantic-Aware Representation Blending for Multi-Label  Image Recognition with Partial Labels](https://arxiv.org/abs/2205.13092) | [⬇️](https://arxiv.org/pdf/2205.13092)
*Tao Pu, Tianshui Chen, Hefeng Wu, Yukai Shi, Zhijing Yang, Liang Lin* 

  Despite achieving impressive progress, current multi-label image recognition
(MLR) algorithms heavily depend on large-scale datasets with complete labels,
making collecting large-scale datasets extremely time-consuming and
labor-intensive. Training the multi-label image recognition models with partial
labels (MLR-PL) is an alternative way, in which merely some labels are known
while others are unknown for each image. However, current MLP-PL algorithms
rely on pre-trained image similarity models or iteratively updating the image
classification models to generate pseudo labels for the unknown labels. Thus,
they depend on a certain amount of annotations and inevitably suffer from
obvious performance drops, especially when the known label proportion is low.
To address this dilemma, we propose a dual-perspective semantic-aware
representation blending (DSRB) that blends multi-granularity category-specific
semantic representation across different images, from instance and prototype
perspective respectively, to transfer information of known labels to complement
unknown labels. Specifically, an instance-perspective representation blending
(IPRB) module is designed to blend the representations of the known labels in
an image with the representations of the corresponding unknown labels in
another image to complement these unknown labels. Meanwhile, a
prototype-perspective representation blending (PPRB) module is introduced to
learn more stable representation prototypes for each category and blends the
representation of unknown labels with the prototypes of corresponding labels,
in a location-sensitive manner, to complement these unknown labels. Extensive
experiments on the MS-COCO, Visual Genome, and Pascal VOC 2007 datasets show
that the proposed DSRB consistently outperforms current state-of-the-art
algorithms on all known label proportion settings.

---------------

### 05 Dec 2023 | [VideoSwap: Customized Video Subject Swapping with Interactive Semantic  Point Correspondence](https://arxiv.org/abs/2312.02087) | [⬇️](https://arxiv.org/pdf/2312.02087)
*Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao,  Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, Kevin Tang* 

  Current diffusion-based video editing primarily focuses on
structure-preserved editing by utilizing various dense correspondences to
ensure temporal consistency and motion alignment. However, these approaches are
often ineffective when the target edit involves a shape change. To embark on
video editing with shape change, we explore customized video subject swapping
in this work, where we aim to replace the main subject in a source video with a
target subject having a distinct identity and potentially different shape. In
contrast to previous methods that rely on dense correspondences, we introduce
the VideoSwap framework that exploits semantic point correspondences, inspired
by our observation that only a small number of semantic points are necessary to
align the subject's motion trajectory and modify its shape. We also introduce
various user-point interactions (\eg, removing points and dragging points) to
address various semantic point correspondence. Extensive experiments
demonstrate state-of-the-art video subject swapping results across a variety of
real-world videos.

---------------
**Date:** 05 Mar 2024

**Title:** CRISPR: Ensemble Model

**Abstract Link:** [https://arxiv.org/abs/2403.03018](https://arxiv.org/abs/2403.03018)

**PDF Link:** [https://arxiv.org/pdf/2403.03018](https://arxiv.org/pdf/2403.03018)

---

**Date:** 10 Jul 2023

**Title:** Model-Driven Engineering for Artificial Intelligence -- A Systematic  Literature Review

**Abstract Link:** [https://arxiv.org/abs/2307.04599](https://arxiv.org/abs/2307.04599)

**PDF Link:** [https://arxiv.org/pdf/2307.04599](https://arxiv.org/pdf/2307.04599)

---

**Date:** 20 Feb 2024

**Title:** dotears: Scalable, consistent DAG estimation using observational and  interventional data

**Abstract Link:** [https://arxiv.org/abs/2305.19215](https://arxiv.org/abs/2305.19215)

**PDF Link:** [https://arxiv.org/pdf/2305.19215](https://arxiv.org/pdf/2305.19215)

---

**Date:** 29 Sep 2023

**Title:** Interpretable neural architecture search and transfer learning for  understanding CRISPR/Cas9 off-target enzymatic reactions

**Abstract Link:** [https://arxiv.org/abs/2305.11917](https://arxiv.org/abs/2305.11917)

**PDF Link:** [https://arxiv.org/pdf/2305.11917](https://arxiv.org/pdf/2305.11917)

---

**Date:** 11 Dec 2023

**Title:** SmartEdit: Exploring Complex Instruction-based Image Editing with  Multimodal Large Language Models

**Abstract Link:** [https://arxiv.org/abs/2312.06739](https://arxiv.org/abs/2312.06739)

**PDF Link:** [https://arxiv.org/pdf/2312.06739](https://arxiv.org/pdf/2312.06739)

---

**Date:** 24 Nov 2023

**Title:** CRISP: Hybrid Structured Sparsity for Class-aware Model Pruning

**Abstract Link:** [https://arxiv.org/abs/2311.14272](https://arxiv.org/abs/2311.14272)

**PDF Link:** [https://arxiv.org/pdf/2311.14272](https://arxiv.org/pdf/2311.14272)

---

**Date:** 29 Jan 2019

**Title:** Revised JNLPBA Corpus: A Revised Version of Biomedical NER Corpus for  Relation Extraction Task

**Abstract Link:** [https://arxiv.org/abs/1901.10219](https://arxiv.org/abs/1901.10219)

**PDF Link:** [https://arxiv.org/pdf/1901.10219](https://arxiv.org/pdf/1901.10219)

---

**Date:** 28 Nov 2011

**Title:** A kernel-based framework for learning graded relations from data

**Abstract Link:** [https://arxiv.org/abs/1111.6473](https://arxiv.org/abs/1111.6473)

**PDF Link:** [https://arxiv.org/pdf/1111.6473](https://arxiv.org/pdf/1111.6473)

---

**Date:** 26 Feb 2019

**Title:** Extending Recurrent Neural Aligner for Streaming End-to-End Speech  Recognition in Mandarin

**Abstract Link:** [https://arxiv.org/abs/1806.06342](https://arxiv.org/abs/1806.06342)

**PDF Link:** [https://arxiv.org/pdf/1806.06342](https://arxiv.org/pdf/1806.06342)

---

**Date:** 16 Nov 2023

**Title:** CRISPR: Eliminating Bias Neurons from an Instruction-following Language  Model

**Abstract Link:** [https://arxiv.org/abs/2311.09627](https://arxiv.org/abs/2311.09627)

**PDF Link:** [https://arxiv.org/pdf/2311.09627](https://arxiv.org/pdf/2311.09627)

---

**Date:** 15 Jan 2014

**Title:** Bounds Arc Consistency for Weighted CSPs

**Abstract Link:** [https://arxiv.org/abs/1401.3481](https://arxiv.org/abs/1401.3481)

**PDF Link:** [https://arxiv.org/pdf/1401.3481](https://arxiv.org/pdf/1401.3481)

---

**Date:** 04 Dec 2023

**Title:** ChemSpaceAL: An Efficient Active Learning Methodology Applied to  Protein-Specific Molecular Generation

**Abstract Link:** [https://arxiv.org/abs/2309.05853](https://arxiv.org/abs/2309.05853)

**PDF Link:** [https://arxiv.org/pdf/2309.05853](https://arxiv.org/pdf/2309.05853)

---

**Date:** 22 Sep 2023

**Title:** FluentEditor: Text-based Speech Editing by Considering Acoustic and  Prosody Consistency

**Abstract Link:** [https://arxiv.org/abs/2309.11725](https://arxiv.org/abs/2309.11725)

**PDF Link:** [https://arxiv.org/pdf/2309.11725](https://arxiv.org/pdf/2309.11725)

---

**Date:** 11 Oct 2016

**Title:** Deep Learning Assessment of Tumor Proliferation in Breast Cancer  Histological Images

**Abstract Link:** [https://arxiv.org/abs/1610.03467](https://arxiv.org/abs/1610.03467)

**PDF Link:** [https://arxiv.org/pdf/1610.03467](https://arxiv.org/pdf/1610.03467)

---

**Date:** 22 Oct 2021

**Title:** Mechanistic Interpretation of Machine Learning Inference: A Fuzzy  Feature Importance Fusion Approach

**Abstract Link:** [https://arxiv.org/abs/2110.11713](https://arxiv.org/abs/2110.11713)

**PDF Link:** [https://arxiv.org/pdf/2110.11713](https://arxiv.org/pdf/2110.11713)

---

**Date:** 12 Jul 2023

**Title:** Improving Chest X-Ray Report Generation by Leveraging Warm Starting

**Abstract Link:** [https://arxiv.org/abs/2201.09405](https://arxiv.org/abs/2201.09405)

**PDF Link:** [https://arxiv.org/pdf/2201.09405](https://arxiv.org/pdf/2201.09405)

---

**Date:** 14 Jun 2023

**Title:** Improving Code-Switching and Named Entity Recognition in ASR with Speech  Editing based Data Augmentation

**Abstract Link:** [https://arxiv.org/abs/2306.08588](https://arxiv.org/abs/2306.08588)

**PDF Link:** [https://arxiv.org/pdf/2306.08588](https://arxiv.org/pdf/2306.08588)

---

**Date:** 22 Oct 2021

**Title:** GeneDisco: A Benchmark for Experimental Design in Drug Discovery

**Abstract Link:** [https://arxiv.org/abs/2110.11875](https://arxiv.org/abs/2110.11875)

**PDF Link:** [https://arxiv.org/pdf/2110.11875](https://arxiv.org/pdf/2110.11875)

---

**Date:** 27 Jan 2024

**Title:** Dual-Perspective Semantic-Aware Representation Blending for Multi-Label  Image Recognition with Partial Labels

**Abstract Link:** [https://arxiv.org/abs/2205.13092](https://arxiv.org/abs/2205.13092)

**PDF Link:** [https://arxiv.org/pdf/2205.13092](https://arxiv.org/pdf/2205.13092)

---

**Date:** 05 Dec 2023

**Title:** VideoSwap: Customized Video Subject Swapping with Interactive Semantic  Point Correspondence

**Abstract Link:** [https://arxiv.org/abs/2312.02087](https://arxiv.org/abs/2312.02087)

**PDF Link:** [https://arxiv.org/pdf/2312.02087](https://arxiv.org/pdf/2312.02087)

---


