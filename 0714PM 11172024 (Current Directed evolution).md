# Query: Current Directed evolution 🧫

## AI Response
### 🔎 Current Directed evolution 🧫



Directed evolution is a method to create new enzymes with desired properties. It is a process that mimics natural evolution in the lab.

The process starts with a population of enzymes that have a desired activity, but not to the extent that is desired. The enzymes are then subjected to a series of rounds of mutagenesis and selection.

Mutagenesis is the process of introducing random mutations into the DNA of the enzymes. This can be done using a variety of methods, such as error-prone PCR, DNA shuffling, or site-directed mutagenesis.

Selection is the process of identifying the enzymes that have the desired activity. This can be done using a variety of methods, such as high-throughput screening, FACS, or yeast display.

The enzymes that have the desired activity are then used to start the next round of mutagenesis and selection. This process is repeated until the desired activity is achieved.

Directed evolution has been used to create enzymes with a wide range of activities, including new catalytic activities, improved stability, and altered substrate specificity. It has also been used to create enzymes that can function in non-natural environments, such as high temperatures, organic solvents, and high salt concentrations.

Directed evolution is a powerful tool for creating new enzymes with desired activities. It is a flexible method that can be adapted to a wide range of enzymes and applications.</s>
# 🩺🔍 Search Results
### 06 Jun 2023 | [Mathematics-assisted directed evolution and protein engineering](https://arxiv.org/abs/2306.04658) | [⬇️](https://arxiv.org/pdf/2306.04658)
*Yuchi Qiu, Guo-Wei Wei* 

  Directed evolution is a molecular biology technique that is transforming
protein engineering by creating proteins with desirable properties and
functions. However, it is experimentally impossible to perform the deep
mutational scanning of the entire protein library due to the enormous
mutational space, which scales as $20^N$ , where N is the number of amino
acids. This has led to the rapid growth of AI-assisted directed evolution
(AIDE) or AI-assisted protein engineering (AIPE) as an emerging research field.
Aided with advanced natural language processing (NLP) techniques, including
long short-term memory, autoencoder, and transformer, sequence-based embeddings
have been dominant approaches in AIDE and AIPE. Persistent Laplacians, an
emerging technique in topological data analysis (TDA), have made
structure-based embeddings a superb option in AIDE and AIPE. We argue that a
class of persistent topological Laplacians (PTLs), including persistent
Laplacians, persistent path Laplacians, persistent sheaf Laplacians, persistent
hypergraph Laplacians, persistent hyperdigraph Laplacians, and evolutionary de
Rham-Hodge theory, can effectively overcome the limitations of the current TDA
and offer a new generation of more powerful TDA approaches. In the general
framework of topological deep learning, mathematics-assisted directed evolution
(MADE) has a great potential for future protein engineering.

---------------

### 25 Oct 2022 | [ODBO: Bayesian Optimization with Search Space Prescreening for Directed  Protein Evolution](https://arxiv.org/abs/2205.09548) | [⬇️](https://arxiv.org/pdf/2205.09548)
*Lixue Cheng, Ziyi Yang, Changyu Hsieh, Benben Liao, Shengyu Zhang* 

  Directed evolution is a versatile technique in protein engineering that
mimics the process of natural selection by iteratively alternating between
mutagenesis and screening in order to search for sequences that optimize a
given property of interest, such as catalytic activity and binding affinity to
a specified target. However, the space of possible proteins is too large to
search exhaustively in the laboratory, and functional proteins are scarce in
the vast sequence space. Machine learning (ML) approaches can accelerate
directed evolution by learning to map protein sequences to functions without
building a detailed model of the underlying physics, chemistry and biological
pathways. Despite the great potentials held by these ML methods, they encounter
severe challenges in identifying the most suitable sequences for a targeted
function. These failures can be attributed to the common practice of adopting a
high-dimensional feature representation for protein sequences and inefficient
search methods. To address these issues, we propose an efficient, experimental
design-oriented closed-loop optimization framework for protein directed
evolution, termed ODBO, which employs a combination of novel low-dimensional
protein encoding strategy and Bayesian optimization enhanced with search space
prescreening via outlier detection. We further design an initial sample
selection strategy to minimize the number of experimental samples for training
ML models. We conduct and report four protein directed evolution experiments
that substantiate the capability of the proposed framework for finding of the
variants with properties of interest. We expect the ODBO framework to greatly
reduce the experimental cost and time cost of directed evolution, and can be
further generalized as a powerful tool for adaptive experimental design in a
broader context.

---------------

### 05 Jun 2022 | [Bandit Theory and Thompson Sampling-Guided Directed Evolution for  Sequence Optimization](https://arxiv.org/abs/2206.02092) | [⬇️](https://arxiv.org/pdf/2206.02092)
*Hui Yuan, Chengzhuo Ni, Huazheng Wang, Xuezhou Zhang, Le Cong, Csaba  Szepesv\'ari, Mengdi Wang* 

  Directed Evolution (DE), a landmark wet-lab method originated in 1960s,
enables discovery of novel protein designs via evolving a population of
candidate sequences. Recent advances in biotechnology has made it possible to
collect high-throughput data, allowing the use of machine learning to map out a
protein's sequence-to-function relation. There is a growing interest in machine
learning-assisted DE for accelerating protein optimization. Yet the theoretical
understanding of DE, as well as the use of machine learning in DE, remains
limited. In this paper, we connect DE with the bandit learning theory and make
a first attempt to study regret minimization in DE. We propose a Thompson
Sampling-guided Directed Evolution (TS-DE) framework for sequence optimization,
where the sequence-to-function mapping is unknown and querying a single value
is subject to costly and noisy measurements. TS-DE updates a posterior of the
function based on collected measurements. It uses a posterior-sampled function
estimate to guide the crossover recombination and mutation steps in DE. In the
case of a linear model, we show that TS-DE enjoys a Bayesian regret of order
$\tilde O(d^{2}\sqrt{MT})$, where $d$ is feature dimension, $M$ is population
size and $T$ is number of rounds. This regret bound is nearly optimal,
confirming that bandit learning can provably accelerate DE. It may have
implications for more general sequence optimization and evolutionary
algorithms.

---------------

### 05 Nov 2021 | [Improving RNA Secondary Structure Design using Deep Reinforcement  Learning](https://arxiv.org/abs/2111.04504) | [⬇️](https://arxiv.org/pdf/2111.04504)
*Alexander Whatley, Zhekun Luo, Xiangru Tang* 

  Rising costs in recent years of developing new drugs and treatments have led
to extensive research in optimization techniques in biomolecular design.
Currently, the most widely used approach in biomolecular design is directed
evolution, which is a greedy hill-climbing algorithm that simulates biological
evolution. In this paper, we propose a new benchmark of applying reinforcement
learning to RNA sequence design, in which the objective function is defined to
be the free energy in the sequence's secondary structure. In addition to
experimenting with the vanilla implementations of each reinforcement learning
algorithm from standard libraries, we analyze variants of each algorithm in
which we modify the algorithm's reward function and tune the model's
hyperparameters. We show results of the ablation analysis that we do for these
algorithms, as well as graphs indicating the algorithm's performance across
batches and its ability to search the possible space of RNA sequences. We find
that our DQN algorithm performs by far the best in this setting, contrasting
with, in which PPO performs the best among all tested algorithms. Our results
should be of interest to those in the biomolecular design community and should
serve as a baseline for future experiments involving machine learning in
molecule design.

---------------

### 08 Jun 2023 | [Multi-level Protein Representation Learning for Blind Mutational Effect  Prediction](https://arxiv.org/abs/2306.04899) | [⬇️](https://arxiv.org/pdf/2306.04899)
*Yang Tan, Bingxin Zhou, Yuanhong Jiang, Yu Guang Wang, Liang Hong* 

  Directed evolution plays an indispensable role in protein engineering that
revises existing protein sequences to attain new or enhanced functions.
Accurately predicting the effects of protein variants necessitates an in-depth
understanding of protein structure and function. Although large self-supervised
language models have demonstrated remarkable performance in zero-shot inference
using only protein sequences, these models inherently do not interpret the
spatial characteristics of protein structures, which are crucial for
comprehending protein folding stability and internal molecular interactions.
This paper introduces a novel pre-training framework that cascades sequential
and geometric analyzers for protein primary and tertiary structures. It guides
mutational directions toward desired traits by simulating natural selection on
wild-type proteins and evaluates the effects of variants based on their fitness
to perform the function. We assess the proposed approach using a public
database and two new databases for a variety of variant effect prediction
tasks, which encompass a diverse set of proteins and assays from different
taxa. The prediction results achieve state-of-the-art performance over other
zero-shot learning methods for both single-site mutations and deep mutations.

---------------

### 18 Mar 2023 | [Protein Sequence Design with Batch Bayesian Optimisation](https://arxiv.org/abs/2303.10429) | [⬇️](https://arxiv.org/pdf/2303.10429)
*Chuanjiao Zong* 

  Protein sequence design is a challenging problem in protein engineering,
which aims to discover novel proteins with useful biological functions.
Directed evolution is a widely-used approach for protein sequence design, which
mimics the evolution cycle in a laboratory environment and conducts an
iterative protocol. However, the burden of laboratory experiments can be
reduced by using machine learning approaches to build a surrogate model of the
protein landscape and conducting in-silico population selection through
model-based fitness prediction. In this paper, we propose a new method based on
Batch Bayesian Optimization (Batch BO), a well-established optimization method,
for protein sequence design. By incorporating Batch BO into the directed
evolution process, our method is able to make more informed decisions about
which sequences to select for artificial evolution, leading to improved
performance and faster convergence. We evaluate our method on a suite of
in-silico protein sequence design tasks and demonstrate substantial improvement
over baseline algorithms.

---------------

### 13 Apr 2023 | [Accurate and Definite Mutational Effect Prediction with Lightweight  Equivariant Graph Neural Networks](https://arxiv.org/abs/2304.08299) | [⬇️](https://arxiv.org/pdf/2304.08299)
*Bingxin Zhou, Outongyi Lv, Kai Yi, Xinye Xiong, Pan Tan, Liang Hong,  Yu Guang Wang* 

  Directed evolution as a widely-used engineering strategy faces obstacles in
finding desired mutants from the massive size of candidate modifications. While
deep learning methods learn protein contexts to establish feasible searching
space, many existing models are computationally demanding and fail to predict
how specific mutational tests will affect a protein's sequence or function.
This research introduces a lightweight graph representation learning scheme
that efficiently analyzes the microenvironment of wild-type proteins and
recommends practical higher-order mutations exclusive to the user-specified
protein and function of interest. Our method enables continuous improvement of
the inference model by limited computational resources and a few hundred
mutational training samples, resulting in accurate prediction of variant
effects that exhibit near-perfect correlation with the ground truth across deep
mutational scanning assays of 19 proteins. With its affordability and
applicability to both computer scientists and biochemical laboratories, our
solution offers a wide range of benefits that make it an ideal choice for the
community.

---------------

### 29 Oct 2018 | [Improving Exploration in Evolution Strategies for Deep Reinforcement  Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/abs/1712.06560) | [⬇️](https://arxiv.org/pdf/1712.06560)
*Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman,  Kenneth O. Stanley, Jeff Clune* 

  Evolution strategies (ES) are a family of black-box optimization algorithms
able to train deep neural networks roughly as well as Q-learning and policy
gradient methods on challenging deep reinforcement learning (RL) problems, but
are much faster (e.g. hours vs. days) because they parallelize better. However,
many RL problems require directed exploration because they have reward
functions that are sparse or deceptive (i.e. contain local optima), and it is
unknown how to encourage such exploration with ES. Here we show that algorithms
that have been invented to promote directed exploration in small-scale evolved
neural networks via populations of exploring agents, specifically novelty
search (NS) and quality diversity (QD) algorithms, can be hybridized with ES to
improve its performance on sparse or deceptive deep RL tasks, while retaining
scalability. Our experiments confirm that the resultant new algorithms, NS-ES
and two QD algorithms, NSR-ES and NSRA-ES, avoid local optima encountered by ES
to achieve higher performance on Atari and simulated robots learning to walk
around a deceptive trap. This paper thus introduces a family of fast, scalable
algorithms for reinforcement learning that are capable of directed exploration.
It also adds this new family of exploration algorithms to the RL toolbox and
raises the interesting possibility that analogous algorithms with multiple
simultaneous paths of exploration might also combine well with existing RL
algorithms outside ES.

---------------

### 12 Jun 2022 | [A Directed-Evolution Method for Sparsification and Compression of Neural  Networks with Application to Object Identification and Segmentation and  considerations of optimal quantization using small number of bits](https://arxiv.org/abs/2206.05859) | [⬇️](https://arxiv.org/pdf/2206.05859)
*Luiz M Franca-Neto* 

  This work introduces Directed-Evolution (DE) method for sparsification of
neural networks, where the relevance of parameters to the network accuracy is
directly assessed and the parameters that produce the least effect on accuracy
when tentatively zeroed are indeed zeroed. DE method avoids a potentially
combinatorial explosion of all possible candidate sets of parameters to be
zeroed in large networks by mimicking evolution in the natural world. DE uses a
distillation context [5]. In this context, the original network is the teacher
and DE evolves the student neural network to the sparsification goal while
maintaining minimal divergence between teacher and student. After the desired
sparsification level is reached in each layer of the network by DE, a variety
of quantization alternatives are used on the surviving parameters to find the
lowest number of bits for their representation with acceptable loss of
accuracy. A procedure to find optimal distribution of quantization levels in
each sparsified layer is presented. Suitable final lossless encoding of the
surviving quantized parameters is used for the final parameter representation.
DE was used in sample of representative neural networks using MNIST,
FashionMNIST and COCO data sets with progressive larger networks. An 80 classes
YOLOv3 with more than 60 million parameters network trained on COCO dataset
reached 90% sparsification and correctly identifies and segments all objects
identified by the original network with more than 80% confidence using 4bit
parameter quantization. Compression between 40x and 80x. It has not escaped the
authors that techniques from different methods can be nested. Once the best
parameter set for sparsification is identified in a cycle of DE, a decision on
zeroing only a sub-set of those parameters can be made using a combination of
criteria like parameter magnitude and Hessian approximations.

---------------

### 23 Feb 2022 | [Using Genetic Programming to Predict and Optimize Protein Function](https://arxiv.org/abs/2202.04039) | [⬇️](https://arxiv.org/pdf/2202.04039)
*Iliya Miralavy, Alexander Bricco, Assaf Gilad and Wolfgang Banzhaf* 

  Protein engineers conventionally use tools such as Directed Evolution to find
new proteins with better functionalities and traits. More recently,
computational techniques and especially machine learning approaches have been
recruited to assist Directed Evolution, showing promising results. In this
paper, we propose POET, a computational Genetic Programming tool based on
evolutionary computation methods to enhance screening and mutagenesis in
Directed Evolution and help protein engineers to find proteins that have better
functionality. As a proof-of-concept we use peptides that generate MRI contrast
detected by the Chemical Exchange Saturation Transfer contrast mechanism. The
evolutionary methods used in POET are described, and the performance of POET in
different epochs of our experiments with Chemical Exchange Saturation Transfer
contrast are studied. Our results indicate that a computational modelling tool
like POET can help to find peptides with 400% better functionality than used
before.

---------------

### 05 Oct 2020 | [AdaLead: A simple and robust adaptive greedy search algorithm for  sequence design](https://arxiv.org/abs/2010.02141) | [⬇️](https://arxiv.org/pdf/2010.02141)
*Sam Sinai, Richard Wang, Alexander Whatley, Stewart Slocum, Elina  Locane, Eric D. Kelsic* 

  Efficient design of biological sequences will have a great impact across many
industrial and healthcare domains. However, discovering improved sequences
requires solving a difficult optimization problem. Traditionally, this
challenge was approached by biologists through a model-free method known as
"directed evolution", the iterative process of random mutation and selection.
As the ability to build models that capture the sequence-to-function map
improves, such models can be used as oracles to screen sequences before running
experiments. In recent years, interest in better algorithms that effectively
use such oracles to outperform model-free approaches has intensified. These
span from approaches based on Bayesian Optimization, to regularized generative
models and adaptations of reinforcement learning. In this work, we implement an
open-source Fitness Landscape EXploration Sandbox (FLEXS:
github.com/samsinai/FLEXS) environment to test and evaluate these algorithms
based on their optimality, consistency, and robustness. Using FLEXS, we develop
an easy-to-implement, scalable, and robust evolutionary greedy algorithm
(AdaLead). Despite its simplicity, we show that AdaLead is a remarkably strong
benchmark that out-competes more complex state of the art approaches in a
variety of biologically motivated sequence design challenges.

---------------

### 05 May 2023 | [Biophysical Cybernetics of Directed Evolution and Eco-evolutionary  Dynamics](https://arxiv.org/abs/2305.03340) | [⬇️](https://arxiv.org/pdf/2305.03340)
*Bryce Allen Bagley* 

  Many major questions in the theory of evolutionary dynamics can in a
meaningful sense be mapped to analyses of stochastic trajectories in game
theoretic contexts. Often the approach is to analyze small numbers of distinct
populations and/or to assume dynamics occur within a regime of population sizes
large enough that deterministic trajectories are an excellent approximation of
reality. The addition of ecological factors, termed "eco-evolutionary
dynamics", further complicates the dynamics and results in many problems which
are intractable or impractically messy for current theoretical methods.
However, an analogous but underexplored approach is to analyze these systems
with an eye primarily towards uncertainty in the models themselves. In the
language of researchers in Reinforcement Learning and adjacent fields, a
Partially Observable Markov Process. Here we introduce a duality which maps the
complexity of accounting for both ecology and individual genotypic/phenotypic
types onto a problem of accounting solely for underlying information-theoretic
computations rather than drawing physical boundaries which do not change the
computations. Armed with this equivalence between computation and the relevant
biophysics, which we term Taak-duality, we attack the problem of "directed
evolution" in the form of a Partially Observable Markov Decision Process. This
provides a tractable case of studying eco-evolutionary trajectories of a highly
general type, and of analyzing questions of potential limits on the efficiency
of evolution in the directed case.

---------------

### 09 Aug 2023 | [Directed differential equation discovery using modified mutation and  cross-over operators](https://arxiv.org/abs/2308.04996) | [⬇️](https://arxiv.org/pdf/2308.04996)
*Elizaveta Ivanchik and Alexander Hvatov* 

  The discovery of equations with knowledge of the process origin is a tempting
prospect. However, most equation discovery tools rely on gradient methods,
which offer limited control over parameters. An alternative approach is the
evolutionary equation discovery, which allows modification of almost every
optimization stage. In this paper, we examine the modifications that can be
introduced into the evolutionary operators of the equation discovery algorithm,
taking inspiration from directed evolution techniques employed in fields such
as chemistry and biology. The resulting approach, dubbed directed equation
discovery, demonstrates a greater ability to converge towards accurate
solutions than the conventional method. To support our findings, we present
experiments based on Burgers', wave, and Korteweg--de Vries equations.

---------------

### 08 Sep 2021 | [Machine learning modeling of family wide enzyme-substrate specificity  screens](https://arxiv.org/abs/2109.03900) | [⬇️](https://arxiv.org/pdf/2109.03900)
*Samuel Goldman, Ria Das, Kevin K. Yang, Connor W. Coley* 

  Biocatalysis is a promising approach to sustainably synthesize
pharmaceuticals, complex natural products, and commodity chemicals at scale.
However, the adoption of biocatalysis is limited by our ability to select
enzymes that will catalyze their natural chemical transformation on non-natural
substrates. While machine learning and in silico directed evolution are
well-posed for this predictive modeling challenge, efforts to date have
primarily aimed to increase activity against a single known substrate, rather
than to identify enzymes capable of acting on new substrates of interest. To
address this need, we curate 6 different high-quality enzyme family screens
from the literature that each measure multiple enzymes against multiple
substrates. We compare machine learning-based compound-protein interaction
(CPI) modeling approaches from the literature used for predicting drug-target
interactions. Surprisingly, comparing these interaction-based models against
collections of independent (single task) enzyme-only or substrate-only models
reveals that current CPI approaches are incapable of learning interactions
between compounds and proteins in the current family level data regime. We
further validate this observation by demonstrating that our no-interaction
baseline can outperform CPI-based models from the literature used to guide the
discovery of kinase inhibitors. Given the high performance of non-interaction
based models, we introduce a new structure-based strategy for pooling residue
representations across a protein sequence. Altogether, this work motivates a
principled path forward in order to build and evaluate meaningful predictive
models for biocatalysis and other drug discovery applications.

---------------

### 21 Feb 2020 | [Accelerating Reinforcement Learning with a  Directional-Gaussian-Smoothing Evolution Strategy](https://arxiv.org/abs/2002.09077) | [⬇️](https://arxiv.org/pdf/2002.09077)
*Jiaxing Zhang, Hoang Tran, Guannan Zhang* 

  Evolution strategy (ES) has been shown great promise in many challenging
reinforcement learning (RL) tasks, rivaling other state-of-the-art deep RL
methods. Yet, there are two limitations in the current ES practice that may
hinder its otherwise further capabilities. First, most current methods rely on
Monte Carlo type gradient estimators to suggest search direction, where the
policy parameter is, in general, randomly sampled. Due to the low accuracy of
such estimators, the RL training may suffer from slow convergence and require
more iterations to reach optimal solution. Secondly, the landscape of reward
functions can be deceptive and contains many local maxima, causing ES
algorithms to prematurely converge and be unable to explore other parts of the
parameter space with potentially greater rewards. In this work, we employ a
Directional Gaussian Smoothing Evolutionary Strategy (DGS-ES) to accelerate RL
training, which is well-suited to address these two challenges with its ability
to i) provide gradient estimates with high accuracy, and ii) find nonlocal
search direction which lays stress on large-scale variation of the reward
function and disregards local fluctuation. Through several benchmark RL tasks
demonstrated herein, we show that DGS-ES is highly scalable, possesses superior
wall-clock time, and achieves competitive reward scores to other popular policy
gradient and ES approaches.

---------------

### 07 Aug 2019 | [ELG: An Event Logic Graph](https://arxiv.org/abs/1907.08015) | [⬇️](https://arxiv.org/pdf/1907.08015)
*Xiao Ding, Zhongyang Li, Ting Liu and Kuo Liao* 

  The evolution and development of events have their own basic principles,
which make events happen sequentially. Therefore, the discovery of such
evolutionary patterns among events are of great value for event prediction,
decision-making and scenario design of dialog systems. However, conventional
knowledge graph mainly focuses on the entities and their relations, which
neglects the real world events. In this paper, we present a novel type of
knowledge base - Event Logic Graph (ELG), which can reveal evolutionary
patterns and development logics of real world events. Specifically, ELG is a
directed cyclic graph, whose nodes are events, and edges stand for the
sequential, causal, conditional or hypernym-hyponym (is-a) relations between
events. We constructed two domain ELG: financial domain ELG, which consists of
more than 1.5 million of event nodes and more than 1.8 million of directed
edges, and travel domain ELG, which consists of about 30 thousand of event
nodes and more than 234 thousand of directed edges. Experimental results show
that ELG is effective for the task of script event prediction.

---------------

### 23 Oct 2019 | [Robot Imitation through Vision, Kinesthetic and Force Features with  Online Adaptation to Changing Environments](https://arxiv.org/abs/1807.09177) | [⬇️](https://arxiv.org/pdf/1807.09177)
*Raul Fernandez-Fernandez, Juan G. Victores, David Estevez and Carlos  Balaguer* 

  Continuous Goal-Directed Actions (CGDA) is a robot imitation framework that
encodes actions as the changes they produce on the environment. While it
presents numerous advantages with respect to other robot imitation frameworks
in terms of generalization and portability, final robot joint trajectories for
the execution of actions are not necessarily encoded within the model. This is
studied as an optimization problem, and the solution is computed through
evolutionary algorithms in simulated environments. Evolutionary algorithms
require a large number of evaluations, which had made the use of these
algorithms in real world applications very challenging. This paper presents
online evolutionary strategies, as a change of paradigm within CGDA execution.
Online evolutionary strategies shift and merge motor execution into the
planning loop. A concrete online evolutionary strategy, Online Evolved
Trajectories (OET), is presented. OET drastically reduces computational times
between motor executions, and enables working in real world dynamic
environments and/or with human collaboration. Its performance has been measured
against Full Trajectory Evolution (FTE) and Incrementally Evolved Trajectories
(IET), obtaining the best overall results. Experimental evaluations are
performed on the TEO full-sized humanoid robot with "paint" and "iron" actions
that together involve vision, kinesthetic and force features.

---------------

### 22 Dec 2022 | [TA2N: Two-Stage Action Alignment Network for Few-shot Action Recognition](https://arxiv.org/abs/2107.04782) | [⬇️](https://arxiv.org/pdf/2107.04782)
*Shuyuan Li, Huabin Liu, Rui Qian, Yuxi Li, John See, Mengjuan Fei,  Xiaoyuan Yu, Weiyao Lin* 

  Few-shot action recognition aims to recognize novel action classes (query)
using just a few samples (support). The majority of current approaches follow
the metric learning paradigm, which learns to compare the similarity between
videos. Recently, it has been observed that directly measuring this similarity
is not ideal since different action instances may show distinctive temporal
distribution, resulting in severe misalignment issues across query and support
videos. In this paper, we arrest this problem from two distinct aspects --
action duration misalignment and action evolution misalignment. We address them
sequentially through a Two-stage Action Alignment Network (TA2N). The first
stage locates the action by learning a temporal affine transform, which warps
each video feature to its action duration while dismissing the
action-irrelevant feature (e.g. background). Next, the second stage coordinates
query feature to match the spatial-temporal action evolution of support by
performing temporally rearrange and spatially offset prediction. Extensive
experiments on benchmark datasets show the potential of the proposed method in
achieving state-of-the-art performance for few-shot action recognition.The code
of this project can be found at https://github.com/R00Kie-Liu/TA2N

---------------

### 16 Jan 2019 | [Evolutionarily-Curated Curriculum Learning for Deep Reinforcement  Learning Agents](https://arxiv.org/abs/1901.05431) | [⬇️](https://arxiv.org/pdf/1901.05431)
*Michael Cerny Green, Benjamin Sergent, Pushyami Shandilya and Vibhor  Kumar* 

  In this paper we propose a new training loop for deep reinforcement learning
agents with an evolutionary generator. Evolutionary procedural content
generation has been used in the creation of maps and levels for games before.
Our system incorporates an evolutionary map generator to construct a training
curriculum that is evolved to maximize loss within the state-of-the-art Double
Dueling Deep Q Network architecture with prioritized replay. We present a
case-study in which we prove the efficacy of our new method on a game with a
discrete, large action space we made called Attackers and Defenders. Our
results demonstrate that training on an evolutionarily-curated curriculum
(directed sampling) of maps both expedites training and improves generalization
when compared to a network trained on an undirected sampling of maps.

---------------

### 27 Feb 2023 | [An algorithmic framework for the optimization of deep neural networks  architectures and hyperparameters](https://arxiv.org/abs/2303.12797) | [⬇️](https://arxiv.org/pdf/2303.12797)
*Julie Keisler (EDF R&D OSIRIS, EDF R&D, CRIStAL), El-Ghazali Talbi  (CRIStAL), Sandra Claudel (EDF R&D OSIRIS, EDF R&D), Gilles Cabriel (EDF R&D  OSIRIS, EDF R&D)* 

  In this paper, we propose an algorithmic framework to automatically generate
efficient deep neural networks and optimize their associated hyperparameters.
The framework is based on evolving directed acyclic graphs (DAGs), defining a
more flexible search space than the existing ones in the literature. It allows
mixtures of different classical operations: convolutions, recurrences and dense
layers, but also more newfangled operations such as self-attention. Based on
this search space we propose neighbourhood and evolution search operators to
optimize both the architecture and hyper-parameters of our networks. These
search operators can be used with any metaheuristic capable of handling mixed
search spaces. We tested our algorithmic framework with an evolutionary
algorithm on a time series prediction benchmark. The results demonstrate that
our framework was able to find models outperforming the established baseline on
numerous datasets.

---------------
**Date:** 06 Jun 2023

**Title:** Mathematics-assisted directed evolution and protein engineering

**Abstract Link:** [https://arxiv.org/abs/2306.04658](https://arxiv.org/abs/2306.04658)

**PDF Link:** [https://arxiv.org/pdf/2306.04658](https://arxiv.org/pdf/2306.04658)

---

**Date:** 25 Oct 2022

**Title:** ODBO: Bayesian Optimization with Search Space Prescreening for Directed  Protein Evolution

**Abstract Link:** [https://arxiv.org/abs/2205.09548](https://arxiv.org/abs/2205.09548)

**PDF Link:** [https://arxiv.org/pdf/2205.09548](https://arxiv.org/pdf/2205.09548)

---

**Date:** 05 Jun 2022

**Title:** Bandit Theory and Thompson Sampling-Guided Directed Evolution for  Sequence Optimization

**Abstract Link:** [https://arxiv.org/abs/2206.02092](https://arxiv.org/abs/2206.02092)

**PDF Link:** [https://arxiv.org/pdf/2206.02092](https://arxiv.org/pdf/2206.02092)

---

**Date:** 05 Nov 2021

**Title:** Improving RNA Secondary Structure Design using Deep Reinforcement  Learning

**Abstract Link:** [https://arxiv.org/abs/2111.04504](https://arxiv.org/abs/2111.04504)

**PDF Link:** [https://arxiv.org/pdf/2111.04504](https://arxiv.org/pdf/2111.04504)

---

**Date:** 08 Jun 2023

**Title:** Multi-level Protein Representation Learning for Blind Mutational Effect  Prediction

**Abstract Link:** [https://arxiv.org/abs/2306.04899](https://arxiv.org/abs/2306.04899)

**PDF Link:** [https://arxiv.org/pdf/2306.04899](https://arxiv.org/pdf/2306.04899)

---

**Date:** 18 Mar 2023

**Title:** Protein Sequence Design with Batch Bayesian Optimisation

**Abstract Link:** [https://arxiv.org/abs/2303.10429](https://arxiv.org/abs/2303.10429)

**PDF Link:** [https://arxiv.org/pdf/2303.10429](https://arxiv.org/pdf/2303.10429)

---

**Date:** 13 Apr 2023

**Title:** Accurate and Definite Mutational Effect Prediction with Lightweight  Equivariant Graph Neural Networks

**Abstract Link:** [https://arxiv.org/abs/2304.08299](https://arxiv.org/abs/2304.08299)

**PDF Link:** [https://arxiv.org/pdf/2304.08299](https://arxiv.org/pdf/2304.08299)

---

**Date:** 29 Oct 2018

**Title:** Improving Exploration in Evolution Strategies for Deep Reinforcement  Learning via a Population of Novelty-Seeking Agents

**Abstract Link:** [https://arxiv.org/abs/1712.06560](https://arxiv.org/abs/1712.06560)

**PDF Link:** [https://arxiv.org/pdf/1712.06560](https://arxiv.org/pdf/1712.06560)

---

**Date:** 12 Jun 2022

**Title:** A Directed-Evolution Method for Sparsification and Compression of Neural  Networks with Application to Object Identification and Segmentation and  considerations of optimal quantization using small number of bits

**Abstract Link:** [https://arxiv.org/abs/2206.05859](https://arxiv.org/abs/2206.05859)

**PDF Link:** [https://arxiv.org/pdf/2206.05859](https://arxiv.org/pdf/2206.05859)

---

**Date:** 23 Feb 2022

**Title:** Using Genetic Programming to Predict and Optimize Protein Function

**Abstract Link:** [https://arxiv.org/abs/2202.04039](https://arxiv.org/abs/2202.04039)

**PDF Link:** [https://arxiv.org/pdf/2202.04039](https://arxiv.org/pdf/2202.04039)

---

**Date:** 05 Oct 2020

**Title:** AdaLead: A simple and robust adaptive greedy search algorithm for  sequence design

**Abstract Link:** [https://arxiv.org/abs/2010.02141](https://arxiv.org/abs/2010.02141)

**PDF Link:** [https://arxiv.org/pdf/2010.02141](https://arxiv.org/pdf/2010.02141)

---

**Date:** 05 May 2023

**Title:** Biophysical Cybernetics of Directed Evolution and Eco-evolutionary  Dynamics

**Abstract Link:** [https://arxiv.org/abs/2305.03340](https://arxiv.org/abs/2305.03340)

**PDF Link:** [https://arxiv.org/pdf/2305.03340](https://arxiv.org/pdf/2305.03340)

---

**Date:** 09 Aug 2023

**Title:** Directed differential equation discovery using modified mutation and  cross-over operators

**Abstract Link:** [https://arxiv.org/abs/2308.04996](https://arxiv.org/abs/2308.04996)

**PDF Link:** [https://arxiv.org/pdf/2308.04996](https://arxiv.org/pdf/2308.04996)

---

**Date:** 08 Sep 2021

**Title:** Machine learning modeling of family wide enzyme-substrate specificity  screens

**Abstract Link:** [https://arxiv.org/abs/2109.03900](https://arxiv.org/abs/2109.03900)

**PDF Link:** [https://arxiv.org/pdf/2109.03900](https://arxiv.org/pdf/2109.03900)

---

**Date:** 21 Feb 2020

**Title:** Accelerating Reinforcement Learning with a  Directional-Gaussian-Smoothing Evolution Strategy

**Abstract Link:** [https://arxiv.org/abs/2002.09077](https://arxiv.org/abs/2002.09077)

**PDF Link:** [https://arxiv.org/pdf/2002.09077](https://arxiv.org/pdf/2002.09077)

---

**Date:** 07 Aug 2019

**Title:** ELG: An Event Logic Graph

**Abstract Link:** [https://arxiv.org/abs/1907.08015](https://arxiv.org/abs/1907.08015)

**PDF Link:** [https://arxiv.org/pdf/1907.08015](https://arxiv.org/pdf/1907.08015)

---

**Date:** 23 Oct 2019

**Title:** Robot Imitation through Vision, Kinesthetic and Force Features with  Online Adaptation to Changing Environments

**Abstract Link:** [https://arxiv.org/abs/1807.09177](https://arxiv.org/abs/1807.09177)

**PDF Link:** [https://arxiv.org/pdf/1807.09177](https://arxiv.org/pdf/1807.09177)

---

**Date:** 22 Dec 2022

**Title:** TA2N: Two-Stage Action Alignment Network for Few-shot Action Recognition

**Abstract Link:** [https://arxiv.org/abs/2107.04782](https://arxiv.org/abs/2107.04782)

**PDF Link:** [https://arxiv.org/pdf/2107.04782](https://arxiv.org/pdf/2107.04782)

---

**Date:** 16 Jan 2019

**Title:** Evolutionarily-Curated Curriculum Learning for Deep Reinforcement  Learning Agents

**Abstract Link:** [https://arxiv.org/abs/1901.05431](https://arxiv.org/abs/1901.05431)

**PDF Link:** [https://arxiv.org/pdf/1901.05431](https://arxiv.org/pdf/1901.05431)

---

**Date:** 27 Feb 2023

**Title:** An algorithmic framework for the optimization of deep neural networks  architectures and hyperparameters

**Abstract Link:** [https://arxiv.org/abs/2303.12797](https://arxiv.org/abs/2303.12797)

**PDF Link:** [https://arxiv.org/pdf/2303.12797](https://arxiv.org/pdf/2303.12797)

---


