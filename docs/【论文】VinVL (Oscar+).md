***

【论文】Zhang P, Li X, Hu X, et al. Vinvl: Revisiting visual representations in vision-language models.（[pdf](https://arxiv.org/pdf/2101.00529.pdf)）

***

## Why is  VinVL

VinVL 的动机很简单，作者发现现有的 VLP 研究主要集中在跨模态交互模型的改进上，但是视觉特征也同样至关重要。于是，VinVL 便着重于改进以对象为中心的视觉表征，提出了一种改进的目标检测模型

改进的 OD（object detection）model 有如下的特点：

- 基于  ResNeXt-152 C4 为 backbone，在 COCO、OpenImages (OI) 、Objects365、Visual Genome (VG) 四个目标检测数据集上进行训练
- 与传统的 OD model 相比（如 X152-FPN），VinVL 的 OD model 可以编码更多样的视觉对象和概念，如下图所示（producing visual representations for 1848 object categories and 524 attribute categories）


<div align=center><img src="https://img-blog.csdnimg.cn/6213d45c18aa4173b840f989518fa8a4.png?x-oss-process=image,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn55m9576KX0FyaWVz,size_20,color_FFFFFF,t_70,g_se,x_16"></div>

结合了改进版 OD model 的 Oscar+ 在 8.85m 的图文对数据集上进行预训练，在如下表所示的下游任务上均取得了不错的表现

<div align=center><img src="https://img-blog.csdnimg.cn/5423385f89fb455c952bce63c3a1de8b.png?x-oss-process=image,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn55m9576KX0FyaWVz,size_20,color_FFFFFF,t_70,g_se,x_16"></div>

## Improving Vision (V) in Vision Language (VL)

之前的 VLP 工作将图像理解模块视为一个黑盒，自 BUTD 以来就不怎么去改进视觉表征。然而，就单目标检测研究来说，取得结果不仅于此，例如

1. 引入了一些更多样化、更丰富、更大的数据集，如 OpenImages 和 Objects 365
2. 在技术上也有许多新的改进，如 pyramid network、one-stage dense prediction 以及 anchor-free detectors 等
3. 利用更强大的 GPU 训练更大的模型

等等都是目标检测领域一些新的结果，而这些结果是可以借鉴到多模态里面的，以获得更好的视觉表征

**Object Detection Pre-training**

如前所述，改进版的 OD model 采四个目标检测数据集进行训练。由于大多数数据集没有属性标注，作者采用了 pre-training + fine-tunning 的方式来训练 OD model，现在四个目标检测数据计算上预训练，然后在 VG 上用附加的属性分支对模型进行微调，使其能够输出对象和属性	

OD model 的预训练数据集构成如下表所示

<div align=center><img src="https://img-blog.csdnimg.cn/32fa7d5d08954739a2221c24e5b1c874.png?x-oss-process=image,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn55m9576KX0FyaWVz,size_20,color_FFFFFF,t_70,g_se,x_16"></div>

另外，关于数据集的采样方法以及OD model 的结构选取、预训练的方式等内容不作为重点，故我们不给出详细介绍，有需要的可以查阅论文

**Injecting attribute information into the model**

作者将属性分支添加到预训练好的 OD model 中，然后在 VG 上微调以注入属性信息（524个类）。由于在 OD  进行了预训练可以很好的获得对象的视觉表征，因此可以将属性损失权重设为一个较大的值，如 1.25

**Efficient region feature extractor for VL tasks**

从特征提取速度的考虑，作者在 VinVL 的 OD model 中做了两点修改，

- 由于视觉对象和属性集更为丰富，经典的 class-aware non-maximal suppression (NMS) post-processing 需要大量时间来移除重叠的边界框，这便使得特征提取过程非常缓慢。为了提高效率，作者用 class-agnostic NMS 取代了class-aware NMS 
- 将 BUTD 中的 dilation = 2 的空洞卷积换成了正常的卷积

这两种替换使得区域特征提取过程快得多，而 VL 下游任务的精度没有任何下降

预训练的 OD  model用作图像理解模块，为VL任务生成 $(\pmb q, \pmb v)$，$\pmb q$ 是检测到的对象名称集合（文本表示），$\pmb v$ 是图像区域特征的集合。每个区域特征表示为 $(\hat v, z)$，其中 $\hat v$ 是 detection head 最后一个线性分类层的输入（$P$-dimensional representation），$z$ 是区域的 $R$-dimensional position encoding（$R=6$）

## OSCAR+ Pre-training

**Pre-training corpus**

作者采用三种类型的 VL 数据集构建 Oscar+ 的预训练语料库

1. image caption datasets，人工标注为 $w$，自动生成的图像标签为 $q$，包含四个数据集 COCO、Conceptual Captions (CC) 、SBU captions 和 flicker30
2. visual QA datas，问题为 $w$，人工标注的答案为 $q$，包括三个数据集  GQA、VQA 和 VG-QAs
3. image tagging dataset，自动生成的 captions 为 $w$，人工标注的 tags 为 $q$，包含 OpenImages 的一个子集，有 1.67M 张图片

<div align=center><img src="https://img-blog.csdnimg.cn/c8af70eebe4341e8922dac97fb918806.png?x-oss-process=image,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn55m9576KX0FyaWVz,size_20,color_FFFFFF,t_70,g_se,x_16"></div>

**Pre-training Objectives**

Oscar+ 的预训练损失函数定义如下
$$
\mathcal L_{Pre-training}=\mathcal L_{MTL}+\mathcal L_{CL3}
$$
其中，$\mathcal L_{MTL}$ 和 Oscar 类似，是文本模态上的 Masked Token Loss；$\mathcal L_{CL3}$ 是 3-way Contrastive Loss，与 Oscar 中的 binary contrastive loss 不同，这里的 contrastive loss 能有效优化 VQA 和 image-text matching

<div align=center><img src="https://img-blog.csdnimg.cn/50bb677c4a5643ba9aab13edb02e4130.png"></div>

$\mathcal L_{CL3}$ 考虑两种类型的训练样本 $x$：（1）以 image caption 和 image tagging data 构建的三元组 { caption, image-tags, image-features }，负样本为 polluted captions $(\pmb w',\pmb q,\pmb v)$ ；（2）以 VQA 数据集构建的 三元组{ question, answer, image-features }，负样本为 polluted answers $(\pmb w,\pmb q',\pmb v)$

在 [CLS] 上接一个全连接层作为三向分类器可以预测三元组的匹配情况，$c=0$ 三元组匹配，$c=1$ 包含 polluted $\pmb w$，$c=2$ 包含 polluted $\pmb q$，3-way Contrastive Loss 定义为如下形式
$$
\mathcal L_{CL3}=-\mathbb E_{(\pmb w, \pmb q,\pmb v;c)\sim\bar{\mathcal D}}\log p(c|f(\pmb w, \pmb q,\pmb v)
$$
其中数据集 $(\pmb w, \pmb q,\pmb v;c)\sim\bar{\mathcal D}$ 包含 50% 的匹配三元组，25% 的 $\pmb w$-polluted 三元组，25% 的 $\pmb q$-polluted 三元组

如下表所示，可以发现 $\pmb q$-polluted 三元组能很好的适应 VQA 的任务目标，但是对于 text-image retrieval task 来说就表现一般，但是总的 3-way Contrastive Loss 还是能够很好的适应两个任务

<div align=center><img src="https://img-blog.csdnimg.cn/2b6fac6375a747f980d612a1fa08b0e9.png?x-oss-process=image,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn55m9576KX0FyaWVz,size_20,color_FFFFFF,t_70,g_se,x_16"></div>

**Pre-trained models**

同样的，文中训练了两个模型变体，Oscar+~B~ 和 Oscar+~L~

## Experiments

下表是 Oscar+ 和其他的 SOTA 方法在下游任务上的比较

<div align=center><img src="https://img-blog.csdnimg.cn/13860114dde747859afaae826018fcae.png?x-oss-process=image,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5aSn55m9576KX0FyaWVz,size_20,color_FFFFFF,t_70,g_se,x_16"></div>


## Reference 

- [CVPR2021《VinVL》用更好的目标检测器提取视觉特征！微软提出VinVL，基于更好的视觉特征，达到更强的多模态性能。](https://zhuanlan.zhihu.com/p/418987007)

