

# ProteinFormer
Protein subcellular localization model based on bioimages and modified transformer.


## Introduction
We propose an end-to-end protein subcellular localization learning model called ProteinFormer, which combines bioimages with Transformer architecture. By incorporating the ResNet model for feature extraction and harnessing the Transformer network for the effective integration of global information, our model achieves outstanding performance on the Cyto\_2017 dataset, surpassing mainstream vision task methods in both single-label and multi-label tasks. To enhance its applicability, we introduce residual learning principles and additional inductive biases into our GL-Protein model when dealing with small sample sizes from the IHC\_2021 dataset. We replace fully connected layers with convolutional layers (ConvFFN) in order to improve its efficiency further. Ablation and comparative analyses demonstrate that our proposed model exhibits superior predictive capabilities compared to convolution-based methods in visual tasks.

## Architecture

* ProteinFormer model
Architecture diagram of ProteinFormer model. The model architecture is mainly divided into feature extraction part, multi-scale fusion part, image feature serialization part, Transformer part. And finally outputs the learned class token (cls) for classification. The structural details of Transformer are shown on the right side of the figure.

![](https://huatu.98youxi.com/markdown/work/uploads/upload_75dbdd6c2945cf62ad9d2dd27059486b.png)


* GL-ProteinFormer model
The architecture and details of the GL-ProteinFormer model. Figure (A) is the architecture diagram of the GL-ProteinFormer model, Figure (B) is the Attention structure diagram modified using residual ideas, Figure (C) is the FFN network, where (a) is the traditional FNN network, (b) is the FNN network after introducing convolutional design.

![](https://huatu.98youxi.com/markdown/work/uploads/upload_924cc71a8cb0327d5c17b9a36acd343f.png)

## Requirements

* Datasets:
1. The Cyto\_2017 dataset \cite{cyto2017Online} was proposed by the International Association for the Advancement of Cytometry at the 2017 Cyto Challenge Conference. 
[CYTO 2017 - 32nd Congress of the International Society for the Advancement of Cytometry | TSNN Trade Show News](https://www.tsnn.com/events/cyto-2017-32nd-congress-international-society-advancement-cytometry)

1.  The IHC\_2021 dataset \cite{ihc-moroki2021databases} was constructed using immunohistochemistry images from the HPA tissue atlas \cite{hpa-tissueuhlen2015tissue}.
* Install
[Install PyTorch](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies.

## Contact
If you have any questions or problems using our tools, please contact us. (Email:[hangs@xtu.edu.cn](mailto:hangs@xtu.edu.cn))

## Authors and acknowledgment
A.X.Y.: model building, designed and performed the experiments, drafted the paper, discussion. L.H.P.: discussion, proofread, revised, and wrote the final version of the paper. H.G.S.: conceived the study, provided expertise, critical guidance and thought leadership, reviewed and revised the manuscript. All authors read and approved the final manuscript. This work was Hunan Provincial Key Research Program(Grant No. 2022WK2009) and Natural Science Foundation of Hunan Province of China Grant 2021JJ30684.

## Citations
<pre style="color: rgb(0, 0, 0); font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; overflow-wrap: break-word; white-space: pre-wrap;">@article{moroki2021databases,<span class="ag-soft-line-break"></span>  title={Databases for technical aspects of immunohistochemistry: 2021 update},<span class="ag-soft-line-break"></span>  author={Moroki, Takayasu and Matsuo, Saori and Hatakeyama, Hirofumi and Hayashi, Seigo and Matsumoto, Izumi and Suzuki, Satoshi and Kotera, Takashi and Kumagai, Kazuyoshi and Ozaki, Kiyokazu},<span class="ag-soft-line-break"></span>  journal={Journal of toxicologic pathology},<span class="ag-soft-line-break"></span>  volume={34},<span class="ag-soft-line-break"></span>  number={2},<span class="ag-soft-line-break"></span>  pages={161--180},<span class="ag-soft-line-break"></span>  year={2021},<span class="ag-soft-line-break"></span>  publisher={Japanese Society of Toxicologic Pathology}<span class="ag-soft-line-break"></span>}</pre>


<pre style="color: rgb(0, 0, 0); font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; overflow-wrap: break-word; white-space: pre-wrap;">@article{uhlen2015tissue,<span class="ag-soft-line-break"></span>  title={Tissue-based map of the human proteome},<span class="ag-soft-line-break"></span>  author={Uhl{\'e}n, Mathias and Fagerberg, Linn and Hallstr{\"o}m, Bj{\"o}rn M and Lindskog, Cecilia and Oksvold, Per and Mardinoglu, Adil and Sivertsson, {\AA}sa and Kampf, Caroline and Sj{\"o}stedt, Evelina and Asplund, Anna and others},<span class="ag-soft-line-break"></span>  journal={Science},<span class="ag-soft-line-break"></span>  volume={347},<span class="ag-soft-line-break"></span>  number={6220},<span class="ag-soft-line-break"></span>  pages={1260419},<span class="ag-soft-line-break"></span>  year={2015},<span class="ag-soft-line-break"></span>  publisher={American Association for the Advancement of Science}<span class="ag-soft-line-break"></span>}</pre>










