from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class StructureMeasure(object):
    def __init__(self):
        self.eps=np.finfo(np.double).eps

    def _Object(self,GT,pred):
        x=np.mean(pred[GT])
        sigma_x=np.std(pred[GT])
        score=2*x/(x*x+1+sigma_x+self.eps)
        return score

    def _S_object(self,GT,pred):
        #compute the similarity of the foreground
        pred_fg=pred.copy()
        pred_fg[~GT]=0
        O_FG=self._Object(GT,pred_fg)

        #compute the similarity of the background
        pred_bg=1-pred.copy()
        pred_bg[GT]=0
        O_BG=self._Object(~GT,pred_bg)

        #combine foreground and background
        u=np.mean(GT)
        Q=u*O_FG+(1-u)*O_BG
        return Q

    def _centroid(self,GT):
        rows,cols=GT.shape
        if np.sum(GT)==0:
            X=round(cols/2)
            Y=round(rows/2)
        else:
            total=np.sum(GT)
            i=range(cols)
            j=range(rows)
            X=int(round(np.sum(np.sum(GT,axis=0)*i)/total))+1
            Y=int(round(np.sum(np.sum(GT,axis=1)*j)/total))+1
        return (X,Y)

    def _divide_GT(self,GT,X,Y):
        rows,cols=GT.shape
        area=rows*cols
        LT=GT[0:Y,0:X]
        RT=GT[0:Y,X:cols]
        LB=GT[Y:rows,0:X]
        RB=GT[Y:rows,X:cols]

        w1=((X)*(Y))/area
        w2=((cols-X)*(Y))/area
        w3=((X)*(rows-Y))/area
        w4=1-w1-w2-w3
        return (LT,RT,LB,RB,w1,w2,w3,w4)

    def _divide_pred(self,pred,X,Y):
        rows, cols = pred.shape
        area = rows * cols
        LT = pred[0:Y, 0:X]
        RT = pred[0:Y, X:cols]
        LB = pred[Y:rows, 0:X]
        RB = pred[Y:rows, X:cols]
        return (LT, RT, LB, RB)

    def _ssim(self,GT,pred):
        rows,cols=GT.shape
        N=rows*cols
        x=np.mean(pred)
        y=np.mean(GT)
        sigma_x2 = np.sum((pred-x)**2)/(N-1+self.eps)
        sigma_y2 = np.sum((GT - y) ** 2) / (N - 1 + self.eps)
        sigma_xy=np.sum((pred-x)*(GT-y))/(N - 1 + self.eps)
        alpha=4*x*y*sigma_xy
        beta=(x**2+y**2)*(sigma_x2+sigma_y2)
        if alpha!=0:
            Q=alpha/(beta+np.finfo(np.double).eps)
        elif alpha==0 and beta==0:
            Q=1.0
        else:
            Q=0
        return Q

    def _S_region(self,GT,pred):
        X,Y=self._centroid(GT)
        GT_LT,GT_RT,GT_LB,GT_RB,w1,w2,w3,w4=self._divide_GT(GT,X,Y)

        Pred_LT,Pred_RT,Pred_LB,Pred_RB=self._divide_pred(pred,X,Y)
        Q1 = self._ssim(GT_LT,Pred_LT)
        Q2 = self._ssim(GT_RT, Pred_RT)
        Q3 = self._ssim(GT_LB, Pred_LB)
        Q4 = self._ssim(GT_RB, Pred_RB)
        Q=w1*Q1+w2*Q2+w3*Q3+w4*Q4
        return Q

    def _minmiax_norm(self,X,ymin=0,ymax=1):
        X = (ymax - ymin) * (X - np.min(X)) / (np.max(X) - np.min(X)) + ymin
        return X

    def _prepare_data(self,GT_path,pred_path):
        pred = np.array(Image.open(pred_path)).astype(np.double)
        GT = np.array(Image.open(GT_path)).astype(np.bool)

        if len(pred.shape)!=2:
            pred=0.2989*pred[:,:,0]+0.5870*pred[:,:,1] + 0.1140*pred[:,:,2]
        if len(GT.shape) != 2:
            GT = GT[:, :, 0]
        #judge channel
        assert len(pred.shape)==2,"Pred should be one channel!"
        assert len(GT.shape)==2,"Ground Truth should be one channel!"
        #normalize
        if np.max(pred)==255:
            pred=(pred/255)
        pred=self._minmiax_norm(pred,0,1)
        return GT,pred

    def __call__(self,GT_path,pred_path):
        GT,pred=self._prepare_data(GT_path,pred_path)
        meanGT=np.mean(GT)
        if meanGT==0:#ground truth is balck
            x=np.mean(pred)
            Q=1.0-x
        elif meanGT==1:#ground truth is white
            x=np.mean(pred)
            Q=x
        else:
            alpha=0.5
            Q=alpha*self._S_object(GT,pred)+(1-alpha)*self._S_region(GT,pred)
            if Q<0:
                Q=0
        return Q



if __name__=="__main__":
    root = os.getcwd()
    GT_dir = os.path.join(root, "demo", "GT")
    Pred_dir = os.path.join(root, "demo", "FG")
    store_dir = os.path.join(root, "demo", "Result")

    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    S_measure=StructureMeasure()
    for filename in os.listdir(GT_dir):
        GT_path=os.path.join(GT_dir,filename)
        id=filename.split('.')[0]
        pred = {}
        pred['0'] = os.path.join(Pred_dir,id+'_DCL.png')
        pred['1'] = os.path.join(Pred_dir,id+'_dhsnet.png')
        pred['2'] = os.path.join(Pred_dir,id+'_DISC.png')
        pred['3'] = os.path.join(Pred_dir,id+'_mc.png')
        pred['4'] = os.path.join(Pred_dir,id+'_MDF.png')
        pred['5'] = os.path.join(Pred_dir,id+'_rfcn.png')
        Sm = np.zeros(6)
        Sm[:] = S_measure(GT_path, pred['0']),S_measure(GT_path, pred['1']),S_measure(GT_path, pred['2']),S_measure(GT_path, pred['3']),S_measure(GT_path, pred['4']),S_measure(GT_path, pred['5']),
        Sm_sort = np.argsort(Sm)

        plt.figure(0,figsize=(7,1))
        #draw GT
        plt.subplot(1, 7, 1)
        plt.title('GT')
        plt.imshow(Image.open(GT_path))
        plt.axis('off')
        for i in range(6):
            plt.subplot(1, 7, 2+i)
            plt.title('S %.4f'%Sm[Sm_sort[i]])
            plt.imshow(Image.open(pred[str(Sm_sort[i])]))
            plt.axis('off')
        plt.savefig(os.path.join(store_dir, filename))
        plt.close(0)