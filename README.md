# 个人数据集上实现物体检测

### 制作数据集
- `python tmp.py` 更改数据集图像文件名
- 使用[labelImg](https://github.com/tzutalin/labelImg)工具制作数据集
- `python tmp2.py` 生成json格式标注
- `python my_data/coco_csv.py` 生成txt格式标注

### 训练模型
- 修改`train.json`并训练