### LOcal COntext based Faster R-CNN

A local context layer is implemented based on Faster R-CNN(see: [py-faster-rcnn code](https://github.com/rbgirshick/py-faster-rcnn)) for detecting small objects more Effectively

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Data preparation](#data-preparation)
5. [Training and Testing](#training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN or LOCO with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the LOCO repository
  ```Shell
  git clone https://github.com/CPFLAME/LOCO.git
  ```

2. We'll call the directory that you cloned LOCO into `LOCO`

   you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    cd LOCO
    git clone https://github.com/rbgirshick/caffe-fast-rcnn.git
    ```

3. Build the Cython modules
    ```Shell
    cd $LOCO/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $LOCO/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
### Data preparation

1. Download the traffic dataset 
    the dataset is modified from Tsinghua-Tencent 100K [a traffic sign dataset](http://cg.cs.tsinghua.edu.cn/traffic-sign/). We treat all traffic signs as one category and transfer it into VOC's annotation  format. For convenience, we call it VOCdevkit2007.

	```Shell
	wget uploading now.
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCdevkit2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit2007/                           # development kit
  	$VOCdevkit2007/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $LOCO/data
    ln -s $VOCdevkit2007 VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

### Training and Testing

1. For training

	you should download pre-trained ImageNet models

	```Shell
	cd $LOCO
	./data/scripts/fetch_imagenet_models.sh
	```

	VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
	ZF was trained at MSRA.

	start training

	```Shell
	cd $LOCO
	./experiments/scripts/faster_rcnn_context.sh [GPU_ID] VGG16 pascal_voc
	```

2. For testing
	We released our pretrained model at [model](http://pan.baidu.com/s/1pKKtKh1), you can download it for testing.

	start testing

	```Shell
	cd $LOCO
	./tools/test_net.py --gpu [GPU_ID] --def models/pascal_voc/VGG16/faster_rcnn_end2end/context_test.prototxt --net $your_model_path --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
	```

### Usage

Trained networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
