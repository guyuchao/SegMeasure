from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
from math import sqrt

class ConfusionMatrixBasedMeasurement:

    def _prepare_data(self,GT_path,pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.bool)
        GT = np.array(Image.open(GT_path)).astype(np.bool)
        if len(pred.shape)!=2:
            pred=pred[:,:,0]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]
        #judge channel
        assert len(pred.shape)==2,"Pred should be one channel!"
        assert len(GT.shape)==2,"Ground Truth should be one channel!"
        return GT,pred

    def IOU(self,GT_path,pred_path):
        GT,pred=self._prepare_data(GT_path,pred_path)
        TP,FP,TN,FN=self._calConfusion(GT,pred)
        Iou=TP/(TP+FN+FP)
        return Iou

    def FbetaMeasure(self,GT_path,pred_path,beta= sqrt(0.3)):
        GT,pred=self._prepare_data(GT_path,pred_path)
        TP,FP,TN,FN=self._calConfusion(GT,pred)
        P=TP/(TP+FP) #precision
        R=TP/(TP+FN) #recall
        Fbeta=(beta**2+1)*P*R/((beta**2)*P+R)
        return Fbeta

    def _calConfusion(self, GT,pred):
        TP=np.sum(pred[GT]==1)
        FP=np.sum(pred[~GT]==1)
        TN=np.sum(pred[~GT]==0)
        FN=np.sum(pred[GT]==0)
        return TP,FP,TN,FN

class EnhancedAlignmentMeasure:
    def __init__(self):
        self.eps=np.finfo(np.double).eps

    def _prepare_data(self,GT_path,pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.bool)
        GT = np.array(Image.open(GT_path)).astype(np.bool)
        if len(pred.shape)!=2:
            pred=pred[:,:,0]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]

        #judge channel
        assert len(pred.shape)==2,"Pred should be one channel!"
        assert len(GT.shape)==2,"Ground Truth should be one channel!"
        return GT,pred

    def _EnhancedAlignmnetTerm(self,align_Matrix):
        enhanced=((align_Matrix+1)**2)/4
        return enhanced

    def _AlignmentTerm(self,dGT,dpred):
        mean_dpred=np.mean(dpred)
        mean_dGT=np.mean(dGT)
        align_dpred=dpred-mean_dpred
        align_dGT=dGT-mean_dGT
        align_matrix=2*(align_dGT*align_dpred)/(align_dGT**2+align_dpred**2+self.eps)
        return align_matrix

    def __call__(self,GT_path,pred_path):
        GT,pred=self._prepare_data(GT_path,pred_path)
        dGT,dpred=GT.astype(np.float64),pred.astype(np.float64)
        if np.sum(GT)==0:#completely black
            enhanced_matrix=1-dpred
        elif np.sum(~GT)==0:
            enhanced_matrix=dpred
        else:
            align_matrix=self._AlignmentTerm(dGT,dpred)
            enhanced_matrix=self._EnhancedAlignmnetTerm(align_matrix)
        rows,cols= GT.shape
        score=np.sum(enhanced_matrix)/(rows*cols-1+self.eps)
        return score


if __name__=="__main__":
    #initialize
    Confusion_measure = ConfusionMatrixBasedMeasurement()
    E_measure = EnhancedAlignmentMeasure()
    #get path
    root=os.getcwd()
    GT_dir=os.path.join(root,"FMDatabase","Imgs")
    Pred_dir=os.path.join(root,"FMDatabase","Saliency")
    store_dir=os.path.join(root,"FMDatabase","result")
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    for filename in os.listdir(GT_dir):
        plt.figure(0,figsize=(8,8))
        idImg=filename.split('.')[0]
        GT_path=os.path.join(GT_dir,filename)
        pred={}
        pred['0']=os.path.join(Pred_dir,str(idImg)+'_A.png')
        pred['1'] = os.path.join(Pred_dir, str(idImg) + '_B.png')
        pred['2'] = os.path.join(Pred_dir, str(idImg) + '_C.png')

        # draw ground truth
        for i in range(0, 12, 4):
            plt.subplot(3, 4, i + 1)
            plt.title('GT')
            plt.imshow(Image.open(GT_path))
            plt.axis('off')

        #Fbeta
        Fbeta=np.zeros(3)
        Fbeta[:]=Confusion_measure.FbetaMeasure(GT_path,pred['0']),Confusion_measure.FbetaMeasure(GT_path,pred['1']),Confusion_measure.FbetaMeasure(GT_path,pred['2'])
        Fbeta_sort=np.argsort(Fbeta)
        #draw Fbeta
        for i in range(3):
            plt.subplot(3, 4, 2+i)
            plt.title('F %.4f'%Fbeta[Fbeta_sort[i]])
            plt.imshow(Image.open(pred[str(Fbeta_sort[i])]))
            plt.axis('off')

        # IOU
        IOU = np.zeros(3)
        IOU[:] = Confusion_measure.IOU(GT_path, pred['0']), Confusion_measure.IOU(GT_path, pred[
            '1']), Confusion_measure.IOU(GT_path, pred['2'])
        IOU_sort = np.argsort(IOU)
        # draw IOU
        for i in range(3):
            plt.subplot(3, 4, 6 + i)
            plt.title('IOU %.4f' % IOU[IOU_sort[i]])
            plt.imshow(Image.open(pred[str(IOU_sort[i])]))
            plt.axis('off')

        # E-measure
        Em = np.zeros(3)
        Em[:] = E_measure(GT_path, pred['0']), E_measure(GT_path, pred['1']), E_measure(GT_path, pred['2'])
        Em_sort = np.argsort(Em)
        # draw IOU
        for i in range(3):
            plt.subplot(3, 4, 10 + i)
            plt.title('E %.4f' % Em[Em_sort[i]])
            plt.imshow(Image.open(pred[str(Em_sort[i])]))
            plt.axis('off')
        plt.savefig(os.path.join(store_dir,filename))
        plt.close(0)