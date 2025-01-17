## News
* DB is included in [WeChat OCR engine](https://mp.weixin.qq.com/s/6IGXof3KWVnN8z1i2YOqJA)
* DB is included in [OpenCV](https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown)
* DB is included in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

# Introduction
This is a PyToch implementation of "Real-time Scene Text Detection with Differentiable Binarization". [This paper](https://arxiv.org/abs/1911.08947) presents a real-time arbitrary-shape scene text detector, achieving the state-of-the-art performance on standard benchmarks.

Part of the code is inherited from [MegReader](https://github.com/Megvii-CSG/MegReader).

## Installation

### Requirements:
- Python3
- PyTorch >= 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name DB -y
  conda activate DB

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/MhLiao/DB.git
  cd DB/

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```
#### 於Windows執行 `python setup.py build_ext --inplace`

以上指令用於 mmdetection 執行前的編譯  
正確執行後會在`DB\assets\ops\dcn\build\lib.win-amd64-3.8`出現以下檔案
```
buildt\lib.win-amd64-3.61\deform_cony_cuda.cp36-win_amd64.pyd
build\lib.win-amd64-3.61\deform_pool_cuda.cp36-win_am
```

##### 常出現的錯誤

###### Error 1
```
soft_renderer/cuda/load_textures_cuda.cpp(24): error C3861: “AT_CHECK”: 找不到标识符
```
###### **Solution**  

>`assets\ops\dcn\src` 中的 `deform_conv_cuda.cpp`和`deform_pool_cuda.cpp`  
將`AT_CHECK` 取代為 → `TORCH_CHECK`  

###### Error 2
```bash
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
```
此處的`['ninja', '-v']`並沒有錯誤，`-v`是編譯所有檔案的指令  
但若檔案內有錯誤造成無法編譯會出現以上報錯  

###### **Solution**  
>`assets\ops\dcn\src` 中的 `deform_conv_cuda_kernel.cu`和`deform_pool_cuda_kernel.cu`  
將`floor` 取代為 → `floorf`  
將`ceil` 取代為 → `ceilf`  
將`round` 取代為 → `roundf`  

##### 其他採坑參考
[Pytorch-DANet编译历程](https://zhuanlan.zhihu.com/p/53418563)

## 常用執行指令
### 訓練
```bash
python train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --num_gpus 1 --epochs 3 --num_workers 1 --batch_size 4
```
`.yaml`為必填參數，後面的 `num_gpus` `epoch` `num_workers` `batch_size`都已經在.yaml裡有預設值但此處能可再次自訂義覆寫  
* `num_gpus` 欲使用的GPU編號，若只有一個GPU則設為0即可  
* `epoch` 欲訓練的回合數  
* `num_workers` 同時多少執行緒並行，只有一個GPU則設為1即可  
* `batch_size` 單次投入的訓練資料數量  

### 驗證
```bash
python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --box_thresh 0.5
```
範例中的`--resume` 非必填參數  
`.yaml`為必填參數，其他可自定義的參數參考如下

```yaml
validation: &validate
        class: ValidationSettings
        data_loaders:
            icdar2015: 
                class: DataLoader
                dataset: ^validate_data
                batch_size: 1
                num_workers: 1
                collect_fn:
                    class: ICDARCollectFN
        visualize: false
        interval: 4500
        exempt: 1
```


### 示範
```
python demo.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml
```
輸出`val_images`資料夾中的圖片預測結果至`.csv`檔案
## Models
此處下載<font color="#f00">預訓練的權重檔案`.pth`</font>  
Download Trained models [Baidu Drive](https://pan.baidu.com/s/1vxcdpOswTK6MxJyPIJlBkA) (download code: p6u3), [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG).
```
  pre-trained-model-synthtext   -- used to finetune models, not for evaluation
  td500_resnet18
  td500_resnet50
  totaltext_resnet18
  totaltext_resnet50
```

## Datasets

* 需自行在`DB`資料夾中新增`datasets`資料夾  
The root of the dataset directory can be ```DB/datasets/```.  

DB預設路徑於`icdar2015`資料夾中
>訓練集 4000張圖片  
>>train_gts 個別標記座標  
>>train_images 圖片檔  
>>train_list.txt 所有訓練集圖片的座標清單  
>>
>驗證集 200張  
>>test_gts   
>>test_images  
>>test_list.txt  
>>
>測試集 1000張 
>>val_images
>>
>測試集(比賽排名用)
>>待公布



* 此處下載<font color="#f00">訓練集(4000張圖)的標記檔案`.txt`</font> 和<font color="#f00">驗證集(500張圖)的標記資料`.txt`與圖片檔`.jpg`</font>  
測試集選用在`TD_TR`資料夾中`TD500`中的`test_images`中的200張圖片  
Download the converted ground-truth and data list [Baidu Drive](https://pan.baidu.com/s/1BPYxcZnLXN87rQKmz9PFYA) (download code: mz0a), [Google Drive](https://drive.google.com/open?id=12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7). The images of each dataset can be obtained from their official website.

* <font color="#f00">訓練集圖片</font>於此[下載](https://tbrain.trendmicro.com.tw/Competitions/Download/13?fileName=TrainDataset_0506.zip)
* <font color="#f00">測試集圖片</font>於此[下載](https://tbrain.trendmicro.com.tw/Competitions/Download/13?fileName=PublicTestDataset.zip)

## Testing
### Prepar dataset
An example of the path of test images: 
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```
The data root directory and the data list file can be defined in ```base_totaltext.yaml```

### Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**

### Demo
Run the model inference with a single image. Here is an example:

```CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --visualize```

The results can be find in `demo_results`.

### Evaluate the performance
Note that we do not provide all the protocols for all benchmarks for simplification. The embedded evaluation protocol in the code is modified from the protocol of ICDAR 2015 dataset while support arbitrary-shape polygons. It almost produces the same results as the pascal evaluation protocol in Total-Text dataset. 

The `img651.jpg` in the test set of Total-Text contains exif info for a 90° rotation thus the gt does not match the image. You should read and re-write this image to get normal results. The converted image is also provided in the dataset links. 

The following command can re-implement the results in the paper:

```
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet50 --polygon --box_thresh 0.6

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet18_deform_thre.yaml --resume path-to-model-directory/td500_resnet18 --box_thresh 0.5

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet50_deform_thre.yaml --resume path-to-model-directory/td500_resnet50 --box_thresh 0.5

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet18_deform_thre.yaml --resume path-to-model-directory/ic15_resnet18 --box_thresh 0.55

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6

# short side 1152, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6
```

The results should be as follows:

|        Model       	| precision 	| recall 	| F-measure 	| precision (paper) 	| recall (paper) 	| F-measure (paper) 	|
|:------------------:	|:---------:	|:------:	|:---------:	|:-----------------:	|:--------------:	|:-----------------:	|
| totaltext-resnet18 	|    88.9   	|  77.6  	|    82.9   	|        88.3       	|      77.9      	|        82.8       	|
| totaltext-resnet50 	|    88.0   	|  81.5  	|    84.6   	|        87.1       	|      82.5      	|        84.7       	|
|   td500-resnet18   	|    86.5   	|  79.4  	|    82.8   	|        90.4       	|      76.3      	|        82.8       	|
|   td500-resnet50   	|    91.1   	|  80.8  	|    85.6   	|        91.5       	|      79.2      	|        84.9       	|
| ic15-resnet18 (736) |    87.7   	|  77.5  	|    82.3   	|        86.8       	|      78.4     	|        82.3       	|
| ic15-resnet50 (736) |    91.3   	|  80.3  	|    85.4   	|        88.2       	|      82.7      	|        85.4       	|
| ic15-resnet50 (1152)|    90.7   	|  84.0  	|    87.2   	|        91.8      	  |      83.2      	|        87.3       	|


```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.

### Evaluate the speed 
Set ```adaptive``` to ```False``` in the yaml file to speedup the inference without decreasing the performance. The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (**Note that the distributed mode is not fully tested. I am not sure whether it can achieves the same performance as non-distributed training.**)

```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

## Improvements
Note that the current implementation is written by pure Python code except for the deformable convolution operator. Thus, the code can be further optimized by some optimization skills, such as [TensorRT](https://github.com/NVIDIA/TensorRT) for the model forward and efficient C++ code for the [post-processing function](https://github.com/MhLiao/DB/blob/d0d855df1c66b002297885a089a18d50a265fa30/structure/representers/seg_detector_representer.py#L26).

Another option to increase speed is to run the model forward and the post-processing algorithm in parallel through a producer-consumer strategy.

Contributions or pull requests are welcome.

## Third-party implementations
* Keras implementation: [xuannianz/DifferentiableBinarization](https://github.com/xuannianz/DifferentiableBinarization)
* DB is included in [OpenCV](https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown)
* DB is included in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## Citing the related works

Please cite the related works in your publications if it helps your research:

     @inproceedings{liao2020real,
      author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
      title={Real-time Scene Text Detection with Differentiable Binarization},
      booktitle={Proc. AAAI},
      year={2020}
    }


    

