import voc_eval as ve

rec,prec,ap = ve.voc_eval('/home/cp/ICIP/tt100k_code/code/tt100k_{:s}_result.txt',
													'/home/cp/ICIP/py-tt100k-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations/{:s}.xml',
													'/home/cp/ICIP/py-tt100k-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt',
													'target',
													'/home/cp/ICIP/py-tt100k-faster-rcnn/data/VOCdevkit2007/annotations_cache',
													0.5,
													True)
print rec,prec,ap
