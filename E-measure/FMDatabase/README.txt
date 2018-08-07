Welcome to the FMDatabase for Binary Foreground Map Evaluation!

=========================================================
The FM Database is contructed for the purpose of evaluate the similarity between 
metric and human ranking. It contains 185 images. Each of the images accompany 
with 3 ranked estimated maps (555 maps in total).

=========================================================
These maps are grouped into two subsets:

1. Imgs: This folder contains 155 ground truth maps.
2. Saliency: This folder contains 555 maps of foreground maps. 
    the ranking order e.g. "1_C", "1_B", "1_A" means that "1_C" rank first, followed by 
    the "1_B", and the last is 1_A. Thus, the human ranking result is 1_C>1_B>1_A. 
=========================================================

By using the dataset you need to cite: 

Enhanced-alignment Measure for Binary Foreground Map Evaluation, Deng-Ping Fan, Cheng Gong, Yang Cao, Bo Ren, Ming-Ming Cheng, Ali Borji, IJCAI 2018 (Oral presentation). 

@inproceedings{Fan2018Enhanced, 
  title={Enhanced-alignment Measure for Binary Foreground Map Evaluation}, 
  author={Deng-Ping Fan, Cheng Gong, Yang Cao, Bo Ren, Ming-Ming Cheng, Ali Borji}, 
  booktitle={IJCAI}, 
  year={2018}, 
  organization={AAAI Press} 
}

@inproceedings{FanStructMeasureICCV17, 
  title={Structure-measure: A New Way to Evaluate Foreground Maps}, 
  author={Deng-Ping Fan and Ming-Ming Cheng and Yun Liu and Tao Li and Ali Borji}, 
  booktitle={IEEE International Conference on Computer Vision},
  pages={4548--4557}, 
  year={2017} 
}

Contact
dengpingfan@mail.nankai.edu.cn

The paper describing the dataset can be downloaded from http://ieeexplore.ieee.org/document/8066351/