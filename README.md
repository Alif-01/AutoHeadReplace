# AutoHeadReplace
自动换头器

## Example
![sample](https://github.com/Alif-01/AutoHeadReplace/blob/master/sample.png)

## Installation
您需要安装`numpy`,`PIL`,`face_recognition`等python库. 安装`face_recognition`的过程中可能会用到`cmake`.

## Usage
使用`replace_faces(image, new_face, feather_radius=5)`进行头部替换.
- `image`, `new_face` 均为PIL中的图像,分别为原图和替换的头.
- 函数返回和`image`相同大小的图像,为换头后的结果.
- `new_face`中应包含一个清晰可识别的头,且最多包含一个.为了保证合成质量,建议使用带透明度通道的图片格式,且除了面部细节的其余部位应为透明(建议去除头发).
- `image`中可包含多个头,应保证每个头清晰可识别且不宜过小(过小或不清晰的头可能会导致识别失败).
- `feather_radius`为边缘羽化半径,默认为5像素,当边缘粗糙时可以调大,当边缘细节丢失时可以调小.

用法示例:
```python
test_img1 = Image.open("yajue.jpg")
test_img2 = Image.open("yajue1.png")

test_img3 = replace_faces(test_img1, test_img2)

test_img3.save("senpai.png")
```
