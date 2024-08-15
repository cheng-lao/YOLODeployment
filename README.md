# YOLOv8 Deployment

实现了一个基础的yolov8 目标检测模型的部署，适用环境为linux.

### 配置openvino环境
~~~bash
git clone https://github.com/openvinotoolkit/openvino

cd openvino
git submodule update --init --recursive

# 如果网络不好的话可以使用gitee 方式拉取
chmod +x scripts/submodule_update_with_gitee.sh
./scripts/submodule_update_with_gitee.sh

# 使用openvino 自带的脚本安装依赖
chmod +x install_build_dependencies.sh
sudo ./install_build_dependencies.sh

# 构建
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make --jobs=$(nproc --all)

# 
make install
~~~
***
### 获取 IR Model(optional)
如果你有一个原始的onnx文件 可以在安装完openvino之后再安装 openvino的python环境
下面展示基于`conda`的python环境安装, **但是项目当中已经提供了IR模型文件和onnx文件这一部分可以跳过.**

[IR模型介绍](https://docs.openvino.ai/2024/documentation/openvino-ir-format.html)
~~~bash
conda env create -n openvino python=3.8 -y

# 激活环境
conda activate openvino

# Update pip
python -m pip install --upgrade pip
# install the Package
python -m pip install openvino

# Verify that the Package Is Installed
python -c "from openvino import Core; print(Core().available_devices)"
~~~
这里有一个[基于virtualenv的openvino环境安装](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-pip.html)，最后的效果是一样的.



接下来可以使用openvino当中的`mo.py`文件来将`onnx`文件模型转换成openvino指定的输入的`IR模型`，[参考链接](http://t.csdnimg.cn/YzTm5).

~~~bash
#在创建的conda环境(or venv)当中
python tools/mo/openvino/tools/mo/mo.py --input_model your_model.pb --output_dir output_dir
~~~

***

### 构建并运行项目

~~~bash
cd openvino

mkdir build

cmake ..

make 

./demo_openvino
~~~


Congratulations :clap:
