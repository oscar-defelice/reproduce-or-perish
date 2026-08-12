# Methods -- extracted from 10.1093/bioinformatics/btab140

2.1 Benchmark datasets
In this study, the used survival analysis benchmark datasets includ-
ing gene expression data, DNA methylation data and clinical data.
The clinical data are included in the main clinical file downloaded
from The Cancer Genome Atlas (TCGA) (Tomczak et al., 2015),
which provides an extensive collection of genomics and clinical out-
come data for large cohorts of patients of more than 30 types of can-
cers. The main files contain 1097 breast cancer patients’ clinical
annotations and information. In our case, two clinical variables are
used: Overall Survival Status (1 if the patient deceased, 0 if he/she is
living at the time of the last follow-up) and Overall Survival
(Months), which represent the number of months between diagnosis
and date of death or last follow-up. In clinical data, patients with
missing follow-up were excluded.
The gene expression data and DNA methylation data of breast
cancer cases pertain to the TCGA dataset of the Broad Institute
GDAC Firehose (Deng et al., 2017). Gene expression information
from mRNA sequence consisted of 20 533 genes. mRNA expression
profiles received the transformation from Illumina HiSeq 2000
RNA-seq readcounts to normalized reads per kilobase per million
(RPKM). This study acquires DNA methylation data as a gene-
related characteristic of 20 106 genes through the selection of the
probe exhibiting a minimal relation to expressing information for
the respective gene. This study removes genes achieving genes
expressing values of 0. Gene expression data, DNA methylation
data and clinical data were merged and filtered to keep only match-
ing samples. This study removes cases with survival months not
recorded or not correctly recorded having negative data. For the lat-
ter reason, among 1097 cases, this study extracts 485 instances that
comprised both mRNA sequencing and DNA methylation informa-
tion. The benchmark dataset including gene data and survival data
was obtained. Table 1 lists the gene- and clinic-related features for
the selected cases.
A challenging point facing the present research is that other sig-
nificant cohorts of breast cancer cases with matched DNA methyla-
tion and gene expression information are lacking. Thus, this study
firstly applies the cross-validation process in the respective steps of
downstream machine learning research, then adopts a second data-
set to validate the efficacy of the proposed method.
Table 1. Gene and clinical characteristics of breast cancer
Characteristics
Summary
Instance no.
485
Gene no.
Methylation
20 106
mRNA
20 533
Survival status
Living
413
Deceased
63
Follow-up (months)
0.03–282.69
Age (years)
Range
26–90
Median
57.23
2602
I.Bichindaritz et al.

2.2 Gene feature extraction
The large number of genes in mRNA and methylation data posed a
challenge to obtaining sufficient statistical power. Recently, a
weighted network mining algorithm termed as local maximum
quasi-clique merging (lmQCM) (Cheng et al., 2017) has been devel-
oped and received favorable results when applied in gene co-
expression research. lmQCM could detect weak quasi-clique mod-
ules in weighted graphs with applications in functional gene cluster
discovery. This algorithm features a greedy approach that uses hier-
archical clustering and does not allow overlap between modules,
Meanwhile, it allows genes to be shared among multiple modules.
This is consistent with the fact that genes often participate in mul-
tiple biological processes. In addition, lmQCM can find smaller
coexpressed gene modules that are often associated with structural
mutations such as copy number variation in cancers. Another well-
known gene clustering algorithm is weighted gene co-expression net-
work
analysis
(WGCNA)
(Langfelder
and
Horvath,
2008).
WGCNA is a powerful technique used to extract co-expressed gene
networks from gene expressions, and is widely used in genomic data
analysis.
In our study, we tested the effectiveness of the methods of
lmQCM and WGCNA respectively. By comparing the effects, we
chose lmQCM as gene feature extraction method. Instead of focus-
ing on individual genes, we firstly use the lmQCM algorithm to clus-
ter genes into coexpressed modules, then summarized each module
as an eigengene. The lmQCM algorithm has four parameters c, t, a,
b. Among these parameters, c is the most influential, as it determines
if a new module can be initiated by setting the weight threshold for
the first edge of the module representing a subnetwork. In the
lmQCM algorithm, the absolute values of the spearman correlation
coefficients between expression profiles of genes are transformed
into weights using a normalization procedure adopted from spectral
clustering. Thus, lmQCM algorithm yields 17 coexpressed gene
modules (features) for methylation data and 116 coexpressed gene
modules for mRNA data. It is worth noting that to avoid overfitting,
we applied gene feature selection methods to the training set and
test set in cross-validation respectively.
2.3 Ordinal Cox model
In survival analysis, prediction of the time duration until a certain
event occurs is the goal of the task being modeled and the death of a
cancer patient is the event of interest in our study (Kourou et al.,
2015). Cancer patients in our study can be divided into two catego-
ries, i.e. censored patients and non-censored patients. For censored
patients, the death events were not observed for them during
the follow-up period, and thus their genuine survival times are lon-
ger than the recorded data; while for non-censored patients, their
recorded survival times are the exact time from initial diagnosis
to death. We use a triplet ðxi; ti; diÞ to represent each obser
vation in survival analysis, where xi is the feature vector, ti is the
observed time and di is the censoring indicator. Here, di ¼ 1 or
di ¼ 0 indicates a non-censored or censored instance, respectively.
The primary goals in survival analysis are estimating the survival
function and hazard function (Wang et al., 2019), both of which can
be used to model the distribution of the event time over the timeline.
Survival function sðtjxÞ represents the probability that the event has
not happened earlier than a specified time t (Lee and Wang, 2003).
We define O as the variable of the true occurrence time for the event
of interest and PrðOÞ is the probabilistic density function (P.D.F.) of
the true event time. So we have,
sðtjxÞ ¼ PrðO  tjxÞ
(1)
By defining the survival function sðtjxÞ as the probability that a
patient will survive after time t, the hazard function that can assess
the instantaneous rate of death is defined as following:
h tjx
ð
Þ ¼ lim
Dt!0
Prðt  O  t þ DtjO  t; xÞ
Dt
(2)
where x ¼ ðx1; x2;   ; xnÞ corresponds to the covariate variable of
dimensionality n. Among the hazards modeling methods, cox
proportional hazard model (Lin et al., 1993), which is built based
on the hypothesis that the hazard ratio between two instances is
time-independent, is defined as:
h tjxÞ ¼ h0 tÞ exp ðhTxÞ


(3)
Here, h0ðtÞ is the baseline hazard, and hTx is called survival
function, in which h ¼ ðh1; h2;   hnÞ can be estimated by minimiz-
ing its corresponding partial likelihood function. The partial likeli-
hood is defined as follows:
l h
ð Þ ¼
Y
i:di¼1
exp ðhðtijxiÞÞ
P
j2RðtiÞ exp ðhðtjjxjÞÞ
(4)
where ti denotes the event time, di is a binary value indicating
whether the event happened or not, and RðtiÞ denotes the set of all
individuals at risk at time ti, which represents the set of patients that
are still at risk before time ti. Therefore, the coefficient vector can be
learned via minimizing the negative partial log-likelihood function
( LCox) of the Cox model, which is defined as following (Sy and
Taylor, 2000):
LCox h
ð Þ ¼ 
Xn
i¼1 di hTxi  log
X
j2RðtiÞ exp ðhTxjÞ


(5)
Although we could use the above Cox model to directly make
survival prediction, it does not take the ordinal survival information
between different cases (e.g. the survival time for case A is longer
than that for case B) into consideration. In the hazard ratio-based
model, the ordinal relationship of the hazard risk between patient
iand patient jcan be easily derived by calculating the ratio (i.e. recij):
recij ¼ hðtjxiÞ
hðtjxjÞ ¼
h0

tÞ exp ðhTxiÞ
h0

tÞ exp ðhTxjÞ
¼ exp ðhT xi  xj
ð
ÞÞ
(6)
In practice, if recij  1, the survival time for patient i should be
shorter than that for patient j, and vice versa. By utilizing the above
ordinal relationship indicated by Cox model, we design a ranking
loss function (LordÞ to capture the ordinal survival information
among different patients as follows:
Lord h
ð Þ ¼ 
Xn
i¼1
X
j6¼iImaxð0; 1  recijÞ
¼ 
Xn
i¼1
X
j6¼iImaxð0; 1  exp ðhTðxi  xjÞÞ
(7)
where I ¼ 1 if the survival time for patient i is shorter than that for
patient j. Otherwise, I ¼ 0.
By combining the Cox negative partial log-likelihood function
LCox with the above ordinal loss Lord, the weighted sum of the losses
can be formulated as a multi-task model. Numerous existing
approaches learning multiple tasks at the same time employ a naive
weighted sum of losses, in which the loss weights are uniform, or
altered in a crude and manual manner. However, the model effect
exhibits extreme sensitivity to weight selecting process. The afore
mentioned weight hyper-parameters can be tuned at high costs.
Thus, a more convenient approach capable of learning the optimal
weights is required. We developed a method to integrate several loss
functions for learning objectives in an adaptive manner.
2.4 Adaptive weighting losses
In this study, we use gene expression and methylation features to
make survival predictions for breast cancer patients. Our main task
is obtaining the training model. The main task has a corresponding
loss Lmain, which can be the expected return loss used for calculating
the policy gradient. The present study employs the Cox negative par-
tial log-likelihood function as the main loss Lmian, i.e. Lmain¼LCox.
To improve data efficiency, besides the main task, one has access to
one or more auxiliary tasks that share some unknown structure with
the main task (Papoudakis et al., 2018). In this study, the ordinal
survival deep network model is employed as an auxiliary task, and
the ordinal loss can be used as auxiliary loss of this auxiliary task,
Integrative survival analysis of breast cancer with gene expression and DNA methylation data
2603

i.e. Laux¼Lord. Our goal is to optimize the main loss Lmian.
However, auxiliary tasks are commonly used to help to learn a good
feature representation. We can combine the main loss with the loss
from the auxiliary tasks as:
Lðh; k1; k2Þ ¼ K1ðk1ÞLmainðhÞ þ K2ðk2ÞLauxðhÞ
(8)
where h is the set of all training model parameters, and K1; K2 are
the weights for the main task and the auxiliary task respectively. Let
KiðkiÞ ¼ eki ði ¼ 1; 2Þ, in which k1, k2 are the weight variables with
an initial value of 0. Under the intuition that modifying h, k1 and
k2 to minimize L will improve Lmain and Laux if the two tasks are
sufficiently related. We propose to modulate the weight variable
k1; k2 at each learning iteration (epoch) t by adding a custom-multi-
loss layer in the deep network. Given that ht is the set of all model
parameters at training step t, and k1t, k2t are the weight variables at
step t, we assume that we update the parameters ht, k1t and k2t using
gradient descent on this combined objective:
htþ1 ¼ ht þ arhtL ht; k1t; k2t
ð
Þ
k1 tþ1
ð
Þ ¼ k1t þ ark1tL ht; k1t; k2t
ð
Þ
k2ðtþ1Þ ¼ k2t þ ark2tL ht; k1t; k2t
ð
Þ
(9)
where a is the gradient step size, and r denotes the gradient of the
loss function L. At each optimization iteration, we can efficiently
approximate the solution to argminðLÞ. The weights are discour-
aged from decreasing too much by the negative exponential func-
tions. The modeling task-dependent weighting can improve the
model’s representation and the performance of each task when com-
pared to separate model trained on each task individually.
2.5 Flowchart of system algorithm
Figure 1 shows the algorithm process of our proposed method.
There are several stages including the gene co-expression cluster
stage, main/auxiliary biLSTM network stage and the COX model
stage etc. In the gene co-expression cluster stage, the feature dimen-
sions of the mRNA and methylation data can be reduced. lmQCM
algorithm is used to cluster genes, and so mRNA and methylation
eigengenes are obtained respectively. The directly concatenated
eigengenes of mRNA and methylation will be main task input fea-
tures for the machine learning network to train the model.
Meanwhile, we also use the concatenated eigengenes as auxiliary
task
input.
In
the
main
task,
multiple
biLSTM
layers,
timeDistributed layers, dropout layers and full connected layers are
used to predict patient survival risk with the negative partial log-
likelihood function, and then the main loss (i.e. Lmain) is obtained.
In the auxiliary task, we use one auxiliary biLSTM layer and one
fully connected layer to obtain the ordinal loss (i.e. auxiliary loss
Laux). We designed a custom multi-loss layer which can combine the
main loss with the auxiliary loss: K1ðk1tÞLmain þ K2ðk2tÞLaux at each
learning iteration t. We use a proposed adaptive optimization iter-
ation method to tune the weight variables (k1t, k2t) of the main and
auxiliary loss. Finally, through iterative training, the deep cox haz-
ard model is built for survival analysis to ensure that the ordinal re-
lationship among the survival time of different patients can be
preserved. We termed this multi-task loss ordinary COX model pro-
cedure as ML_ordCOX.
2.6 Evaluation indexes
This study assesses the performance of the developed approach and
other comparing method using Concordance index (C-index). C-
index quantifies the fraction of all pairs of cases with predicted sur-
vival times ordered in a correct manner as:
C  index ¼ 1
k
Xm
i¼1
X
j:ti < tjIðFðxiÞ < FðxjÞÞ
(10)
where k denotes the set of validly orderable pairs when ti < tj;
k represents the number of comparable pairs among them; FðxÞ is
the prediction of survival time; I is the indicator function of whether
the condition in parentheses is satisfied or not. C-index gives prob-
ability. In terms of a random individual pair, the predicted survival
time of the two individuals is in the same order as their actual sur-
vival time. Since the C-index is determined only by variations in the
predicted results, it is very useful for evaluating proportional hazard
models. Because the order of proportional-risk models doesn’t
change over time. Therefore, we were able to use relative risk func-
tions rather than measures used to predict survival time.