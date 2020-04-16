# PaddleHub人脸检测示例

本示例 [https://aistudio.baidu.com/aistudio/projectdetail/403504](https://aistudio.baidu.com/aistudio/projectdetail/403504) 利用Ultra-Light-Fast-Generic-Face-Detector-1MB模型完成人脸检测。该模型是针对边缘计算设备或低算力设备(如用ARM推理)设计的实时超轻量级通用人脸检测模型，可以在低算力设备中如用ARM进行实时的通用场景的人脸检测推理。

[toc]

## 环境安装

CUDA和cuDNN的安装可以查看我写过的一篇文章

[https://blog.csdn.net/CANGYE0504/article/details/104455394](https://blog.csdn.net/CANGYE0504/article/details/104455394)

安装完成后我们接下来安装飞桨

飞桨各个版本的安装可以参考

[https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)

我这边系统是`ubuntu 19.10`，`CUDA 10.1`

当然，首先建议在虚拟环境下安装，虚拟环境的配置可以参考

[https://blog.csdn.net/CANGYE0504/article/details/105158753](https://blog.csdn.net/CANGYE0504/article/details/105158753)

安装好虚拟环境后，我们新建一个虚拟环境`paddleproject`

```bash
mkvirtualenv paddleproject
```

```bash
> $ mkvirtualenv paddleproject                                                 
created virtual environment CPython3.7.5.final.0-64 in 219ms
  creator CPython3Posix(dest=/home/mrcangye/.virtualenvs/paddleproject, clear=False, global=False)
  seeder FromAppData(download=False, pip=latest, setuptools=latest, wheel=latest, via=copy, app_data_dir=/home/mrcangye/.local/share/virtualenv/seed-app-data/v1.0.1)
  activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator
virtualenvwrapper.user_scripts creating /home/mrcangye/.virtualenvs/paddleproject/bin/predeactivate
virtualenvwrapper.user_scripts creating /home/mrcangye/.virtualenvs/paddleproject/bin/postdeactivate
virtualenvwrapper.user_scripts creating /home/mrcangye/.virtualenvs/paddleproject/bin/preactivate
virtualenvwrapper.user_scripts creating /home/mrcangye/.virtualenvs/paddleproject/bin/postactivate
virtualenvwrapper.user_scripts creating /home/mrcangye/.virtualenvs/paddleproject/bin/get_env_details
```

接下来在虚拟环境中安装飞桨

```bash
python3 -m pip install paddlepaddle-gpu==1.7.1.post107 -i https://mirror.baidu.com/pypi/simple
```

安装完成后我们测试安装情况

首先在虚拟环境终端下输入python3,再输入以下代码

```bash
(paddleproject) > $ python3                                                    
Python 3.7.5 (default, Nov 20 2019, 09:21:52) 
[GCC 9.2.1 20191008] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import paddle.fluid
>>> paddle.fluid.install_check.run_check()
```

如果最后一行显示：

```bash
Your Paddle works well on MUTIPLE GPU or CPU.
Your Paddle is installed successfully! Let's start deep Learning with Paddle now
```

这样就安装成功啦

接下来我们安装人脸检测需要的包

```bash
pip3 install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这个包安装完成后，我们所有必须的环境就已经安装完毕。

## 图片选择

以最近比较火的黑人专业团队为待预测图片

![](https://image.cangye.me/2020/04/16/test.jpg)

这个图片名为`test.jpg`

如果你这边有好多图片，那么我们可以在目录下新建一个`txt`文件，将图片地址一行一行写上去，再让`python`读取就可以啦。例如以下情况。

```bash
# test.txt
./test.jpg
```

## 程序设计

毕竟是个项目，我们要新建一个项目文件，不如叫`paddlecv`

```bash
mkdir paddlecv
```

然后再把图像和`txt`移到文件里

接下来我们编写程序`paddlecv.py`

```python
#paddlecv.py
import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 打开文件写的图像地址，读取图像
with open('test.txt', 'r') as f:
    test_img_path=[]
    for line in f:
        test_img_path.append(line.strip())
    print(test_img_path)
# 模型加载
module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
input_dict = {"image": test_img_path}

# 模型处理
results = module.face_detection(data=input_dict,visualization=True)
for result in results:
    print(result)


# 预测结果展示
img = mpimg.imread("face_detector_640_predict_output/test.jpg")
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()
```

这个程序写好之后就可以运行，运行结果就是酱紫的：

![](https://image.cangye.me/2020/04/16/test_face_detection.jpg)

毕竟人家是专业团队……