# Sub-Band Based Attention for Robust Polyp Segmentation
This repo is the official implementation for the paper **"Sub-Band Based Attention for Robust Polyp Segmentation"** at IJCAI-23.

In this work, we propose a Sub-Band based Attention (SBA) module which uniformly adopts either the high or middle sub-bands of the encoder features to boost the decoder features and thus concretely improve the feature discrimination. We also introduce a Transformer Attended Convolution (TAC) module as the main encoder block which takes the Transformer features to boost the CNN features with stronger long-range object contexts. The combination of SBA and TAC leads to a novel polyp segmentation framework, SBA-Net. It adopts TAC to effectively obtain encoded features which also input to SBA, so that efficient sub-bands based attention maps can be generated for progressively decoding the bottleneck features.  
![network](subnet.png)


## Recommended Environment: 

 Python 3.6 
 
 PyTorch 1.10.2 
 
 torchvision 0.11.3 
 
 numpy 1.19.5 
 
 scipy 1.5.4 
 
 tqdm 4.64.1 
 

## Preparation 
 Please download [training and testing datasets](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view)  and move them into ./dataset/. 
 
 Also please download [PVT v2 weights](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV) and move it into ./pretrained/. 
 

## Checkpoint 

You may also download our [pretrained weights](https://drive.google.com/file/d/1SmRmelNBtToW3abCUi-lG6XUaJMrAM1c/view?usp=sharing) and move it into ./snapshots/ for test. 
 

## Training and Testing 
### Training 
 The training results can be found in ./snapshots/. You can try the following code for training: 
 
python Train.py 
 

### Testing 

The testing results can be found in a new fold ./results/. You can try the following code for testing: 

python Test.py

## Citation
Please cite our paper "Sub-Band Based Attention for Robust Polyp Segmentation" on this work if it is useful for your work. Thank you.

@inproceedings{fang2023SBANet,

  title={Sub-Band Based Attention for Robust Polyp Segmentation},
  
  author={Xianyong Fang and Yuqing Shi and Qingqing Guo and Linbo Wang and Zhengyi Liu},
  
  booktitle={Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI-23)},
  
  pages={736--744},
  
  year={2023}
  
}
