# Arcface : Face Recognition System (Jetson Nano, flutter, firebase) #

<br>

## Demonstration ##
<p align="center">
<img src="https://github.com/ejrtks1020/face-racognition-system-jetson_nano/assets/49896157/93815310-9f7b-4aff-b48a-804b7bbb0d44">
</p>


## Environments ##

* OS : macOS Monterey
* Memory : 16GB
* CPU : Apple M1 Pro
* Python : 3.9.12
* pip : 21.2.4
* anaconda : 4.12.0

## model information ##
**insightface buffalo_l model pack**

* Face Detection Model : Retinaface
* Backbone : Resnet50
* Loss function : ArcFace(Additive Angular Margin Loss)
* Embedded System : Jetson Nano

## set up ##
* create conda environment **(python==3.9.12)**
* git clone this repository
* pip install -r requirements.txt
* 다음 패키지를 **순서대로** 설치 (M1 Mac 기준)
<pre><code>
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
conda install protobuf
conda install python-flatbuffers
pip install insightface
conda install onnxruntime
</code></pre>


