## Facenet+Retinaface：人脸识别模型在Keras当中的实现
---

## 目录
1. [注意事项 Attention](#注意事项)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [参考资料 Reference](#Reference)

## 注意事项
该库中包含了两个网络，分别是retinaface和facenet。二者使用不同的权值。
在使用网络时一定要注意权值的选择，以及主干与权值的匹配。

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 文件下载
预测所需的权值文件可以在百度云下载。     
链接: https://pan.baidu.com/s/1byskhV594bK9b0eHONjF2g 提取码: tn8y  

## 预测步骤
1. 本项目自带主干为mobilenet的retinaface模型与facenet模型。可以直接运行，如果想要使用主干为resnet50的retinafa和主干为inception_resnetv1的facenet模型需要。
2. 在retinaface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件。  
```python
_defaults = {
    "retinaface_model_path" : 'model_data/retinaface_mobilenet025.h5',
    #-----------------------------------#
    #   可选retinaface_backbone有
    #   mobilenet和resnet50
    #-----------------------------------#
    "retinaface_backbone"   : "mobilenet",
    "confidence"            : 0.5,
    "iou"                   : 0.3,
    #----------------------------------------------------------------------#
    #   是否需要进行图像大小限制。
    #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
    #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
    #   keras代码中主干为mobilenet时存在小bug，当输入图像的宽高不为32的倍数
    #   会导致检测结果偏差，主干为resnet50不存在此问题。
    #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
    #----------------------------------------------------------------------#
    "retinaface_input_shape": [640, 640, 3],
    "letterbox_image"       : True,

    "facenet_model_path"    : 'model_data/facenet_mobilenet.h5',
    #-----------------------------------#
    #   可选facenet_backbone有
    #   mobilenet和inception_resnetv1
    #-----------------------------------#
    "facenet_backbone"      : "mobilenet",
    "facenet_input_shape"   : [160,160,3],
    "facenet_threhold"      : 0.9,
}
```
3. 运行encoding.py，对face_dataset里面的图片进行编码，face_dataset的命名规则为XXX_1.jpg、XXX_2.jpg。最终在model_data文件夹下生成对应的数据库人脸编码数据文件。
4. 运行predict.py，输入下述文字，可直接预测。
```python
img/zhangxueyou.jpg
```  
5. 在predict.py里面进行设置可以进行fps测试和video视频检测。  


## Reference
https://github.com/biubug6/Pytorch_Retinaface

