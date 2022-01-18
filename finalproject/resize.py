# import Image
# infile = 'C:\\users\\Lucky\\PycharmProjects\\stylegan-master\\datasets\\faces'
# outfile = 'C:\\users\\Lucky\\PycharmProjects\\stylegan-master\\datasets\\new_faces'
# # infile = 'D:\\original_img.jpg'
# # outfile = 'D:\\adjust_img.jpg'
# im = Image.open(infile)
# (x,y) = im.size #read image size
# x_s = 64 #define standard width
# #y_s = y * x_s / x #calc height based on standard width
# y_s = 64
# out = im.resize((x_s,y_s),Image.ANTIALIAS) #resize image with high-quality
# out.save(outfile)
#
# print ('original size: ',x,y)
# print ('adjust size: ',x_s,y_s)

from PIL import Image
import glob, os
root = os.path.abspath("C:/Users/Lucky/PycharmProjects/stylegan-master/datasets/faces")
img_path = glob.glob(os.path.join(root, '*'))
#img_path = glob.glob("C:/Users/Lucky/PycharmProjects/stylegan-master/datasets/face/*.jpg")
#path_save = "C:/Users/Lucky/PycharmProjects/stylegan-master/datasets/new_faces"
for file in img_path:
    name = os.path.join(file)
    im = Image.open(file)
    im.thumbnail((64,64))
    print(im.format, im.size, im.mode)
    im.save(name,'JPEG')

# img_path = '/Users/corecode/Desktop/Camera/'
# path_save = '/Users/corecode/Desktop/newCamera/'
#
# for i in range(1,260):
#   im = Image.open(img_path+str(i)+'photo.jpg')
#   im.thumbnail((800,800))
#   print("i:"+str(i))
#   im.save(str(i)+'.jpg','JPEG')