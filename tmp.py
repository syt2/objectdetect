import cv2 as cv
import os
import shutil
# ==================可能需要修改的地方=====================================#
# g_root_path = "/Users/sanyito/Downloads/coco/images"
# os.chdir(g_root_path)  # 更改工作路径到图片根目录
org_path = "/Users/sanyito/Downloads/coco/images/tmp/"  # 原图片目录
dst_path = "/Users/sanyito/Downloads/coco/images/"  # 目标图片目录
img_cnt = 0
# ==================================================================#

file_list = os.listdir(org_path)
print(file_list)
if os.path.exists(dst_path) is False:
    os.makedirs(dst_path)
for idx, file in enumerate(file_list):
    shutil.copyfile(org_path+file, dst_path+"%06d.jpg" % img_cnt)
    # img = cv.imread(org_path + file)
    # # img=cv.resize(img,(512,512))
    # img_name = os.path.join(dst_path, "%06d.jpg" % img_cnt)
    # cv.imwrite(img_name, img)
    img_cnt += 1