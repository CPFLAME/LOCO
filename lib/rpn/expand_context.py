__author__ = 'BK'

import caffe
import copy
DEBUG = False

class expandcontext(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        top[0].reshape(1, 5)

        top[1].reshape(1, 5)

        top[2].reshape(1, 5)


    def forward(self, bottom, top):

        rois = bottom[1].data

        rois_x=copy.deepcopy(rois)
        rois_y=copy.deepcopy(rois)
        # print 'look here ',bottom[0].data[0]
        for i in range(rois.shape[0]):
                rois_x[i][1]=max(0,2*rois[i][1]-rois[i][3])
                rois_x[i][3]=min(bottom[0].data[0][0],2*rois[i][3]-rois[i][1])

                rois_y[i][2]=max(0,2*rois[i][2]-rois[i][4])
                rois_y[i][4]=min(bottom[0].data[0][1],2*rois[i][4]-rois[i][2])

        #print 'rois ',rois[0]
        #print 'rois_x ',rois_x[0]
        #print 'rois_y ',rois_y[0]

        top[0].reshape(*rois.shape)
        top[0].data[...] = rois
        top[1].reshape(*rois.shape)
        top[1].data[...] = rois_x
        top[2].reshape(*rois.shape)
        top[2].data[...] = rois_y
	#print "rois",rois
	#print "rois_w",rois_x
	#print "rois_h",rois_y

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

