# Efficient Spectral-Spatial Squeeze and Excitation Residual Bag-of-Features Learning for Hyperspectral Imagery.

## Description

The BilinearNet comprises 3D and 2D fused CNN model for HSI classification. The 3D CNN extracts the joint spatial-spectral features, whereas the 2D-CNN extracts the original spatial features. Both 3D and 2D CNNs are trained simultaneously in a bilinear training fashion. The final features of both networks are fused using a multilayer perceptron which is also trained jointly. The BilinearNet model is completely end-to-end trainable. 

## Model

<img src="figure/S3EResBoF.jpg"/>

Fig.1 The spectral-spatial squeeze-and-excitation residual Bag-of-feature~($S3EResBoF$) learning for HSI classification framework. The first step is sample extraction, where $S\times{S}\times{B}$ sized sample is extracted from a neighborhood window  centered around the target pixel.  Once  samples are  extracted  from  raw  HSI,  they are put through the $S3EResBoF$ to extract deep spectral-spatial features for calculate of classification scores..

## Prerequisites

- [Anaconda 2.7](https://www.anaconda.com/download/#linux)
- [Tensorflow 1.3](https://github.com/tensorflow/tensorflow/tree/r1.3)
- [Keras 2.0](https://github.com/fchollet/keras)

## Results

### Indian Pines (IP) dataset

<img src="figure/IP-FC.jpg" width="200" height="200"/> <img src="figure/IP-GT.jpg" width="200" height="200"/> <img src="figure/IP-Pr.jpg" width="200" height="200"/> <img src="figure/IP_legend.jpg" width="250" height="150"/>

Fig.2  The IN dataset classification result (Overall Accuracy 99.81%) of BilinearNet using 30% samples for training. (a) False color image. (b) Ground truth labels. (c) Classification map. (d) Class legend. 

### University of Pavia (UP) dataset

<img src="figure/UP-FC.jpg"/> <img src="figure/UP-GT.jpg"/> <img src="figure/UP-Pr.jpg"/> <img src="figure/UP_legend.jpg" width="200" height="100"/>

Fig.3  The UP dataset classification result (Overall Accuracy 99.99%) of BilinearNet using 30% samples for training. (a) False color image. (b) Ground truth labels. (c) Classification map. (d) Class legend.

### Salinas Scene (SS) dataset

<img src="figure/SA-FC.jpg"/> <img src="figure/SA-GT.jpg"/> <img src="figure/SA-Pr.jpg"/> <img src="figure/SA_legend.jpg" width="300" height="150"/>

Fig.4  The SS dataset classification result (Overall Accuracy 100%) of BilinearNet using 30% samples for training. (a) False color image. (b) Ground truth labels. (c) Classification map. (d) Class legend.

### Confusion matrices
<img src="figure/IP-3D.jpg" width="280" height="280"/><img src="figure/UP-3D.jpg" width="280" height="280"/><img src="figure/SA-3D.jpg" width="280" height="280"/> 

Fig.5  The confusion matrices for the datasets. (a) Indian Pines. (b) University of Pavia. (c) Salinas Scene. 

#### Detailed results can be found in the [Supplementary Material]

## Citation

If you use this code in your research, we would appreciate a citation to the original [paper]:

    @article{roy2019bilinearnet,
     title={Efficient Spectral-Spatial Squeeze and Excitation Residual Bag-of-Features Learning for Hyperspectral Imagery},
     author={Roy, Swalpa Kumar and Chatterjee, Subhrasankar and Dubey,  and Chaudhuri, Bidyut B.},
     journal={},
     year={2019}
     }


## Acknowledgement

Part of this code is from a implementation of Classification of HSI using CNN by [Konstantinos Fokeas](https://github.com/KonstantinosF/Classification-of-Hyperspectral-Image).

## License

Copyright (c) 2019 Subhrasankar Chatterjee. Released under the MIT License. See [LICENSE](LICENSE) for details.
