# ADNet-tensorflow

- ADNet implemented by Tensorflow

- I implement it almost same with original flows.

- If fact, the authors's code is something wrong. it is not same with his paper. So i recommend not to use it.

## Implements performance
-Environments : Intel® Core™ i7-7700 CPU @ 3.60GHz × 8, GeForce GTX 1070

-Prec. (20px) : (Paper :  85.1) (this code :  79.4)

-IOU(AUC)     : (Paper : 0.635) (this code : 0.601)

-FPS          : (Paper : 15FPS) (this code : 18FPS)

## requirement
-Tensorflow

-Scipy

-OpenCV

## Run
```
$ python tracker.py
``` 
## Dataset
the codes is 

## References
- Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning (CVPR2017) : http://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf
- ADNet Implmentation in Matlab : https://github.com/hellbell/ADNet/
