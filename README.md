# 哔哩哔哩课程下载器 / bilibili cheeses downloader 下载你已购买的或免费的 bilibili 课程。
## 本项目在https://github.com/Misaka10571/bilibili-cheese-downloader 上改进而来
使用方式：下载 bdownloader.py 或下载 release 中的 exe 文件后运行即可。手机 app 扫码完成登录，随后输入你已拥有的课程 ID 即可完成下载操作。  
注意：通过 ffmpeg 进行音视频合成操作，所以请提前配置好环境变量或将其复制到主程序同级目录下。  
基本是 ai 搓的，所以问题有很多，有空再改改。  
所需依赖；  
### python=3.9
### pip3 install tqdm bilibili-api-python aiohttp  
点个 star 谢谢喵，爱你喵。

### 2.0版本实现了自定义下载任意课程节数
### 3.0版本能将原本H.264编码的课程转换为压缩效率更高的H.265编码，并且提供帧率的修改，实现对文件体积的进一步压缩
