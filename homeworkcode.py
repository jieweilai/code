from PIL import Image as im 
from matplotlib import pyplot as plt 
from collections import Counter 
import numpy as np
import os

def cut_image(image,n,m):
    width, height = image.size
    item_width = int(width / n)
    item_height=int(height/m)
    box_list = []
    for i in range(0,n):
        for j in range(0,m):
            box = (j*item_width,i*item_height,(j+1)*item_width,(i+1)*item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]  
    return image_list

def save_images(image_list):
    index = 0
    for image in image_list:
        plt.title(index)
        plt.imshow(image)
        plt.show()
        image.save('/content/subimages/'+str(index) + '.png', 'PNG')
        index += 1

def create_histogram(r,g,b,path):
  ar=np.array(r).flatten()
  plt.hist(ar, bins=256, facecolor='r',edgecolor='r')
  ag=np.array(g).flatten()
  plt.hist(ag, bins=256,  facecolor='g',edgecolor='g')
  ab=np.array(b).flatten()
  plt.hist(ab, bins=256, facecolor='b',edgecolor='b')
  plt.savefig(path)
  plt.show()

def fitter(arr):
  arr=np.array(arr).flatten()
  new_arr=np.array(arr)
  m=np.where(arr>=max(arr)*0.8)
  n=np.where(arr<=max(arr)*0.2)
  for i in m:
    new_arr[i]=max(arr)*0.8
  for i in n:
    new_arr[i]=0
  return new_arr

def p(arr):
  counter=Counter(arr)#统计各个像素点的个数
  p=np.zeros(256)#计算概率密度
  for i in counter.keys():
    p[i]=counter[i]/sum(counter.values())
  return p
  
def t(p):
  t=np.zeros(256)
  for i in range(0,256):
    for j in range(0,i):
      t[i]=p[j]+t[i]
  for i in range(0,256):
    t[i]=t[i]*255+0.5
  return t

def recreate(arr,t):
  shape=np.array(arr).shape
  arr=np.array(arr).flatten()
  new_img=np.zeros(len(arr))
  for i in range(0,len(arr)):
    new_img[i]=int(t[arr[i]])
  return im.fromarray(new_img.reshape(shape)).convert('L')

def getk(path,n):
  img=im.open(path)
  r,g,b,a=img.split()
  r,g,b=np.array(r).flatten(),np.array(g).flatten(),np.array(b).flatten()
  return r[n],g[n],b[n]

def getT(path1,path2,n):#path1 near_image  path2 aim_image n index
  img=im.open(path1)
  r,g,b,a=img.split()
  fr,fg,fb=fitter(r),fitter(g),fitter(b)
  Pr,Pg,Pb=p(fr),p(fg),p(fb)
  Tr,Tg,Tb=t(Pr),t(Pg),t(Pb)
  k=getk(path2,n)
  return Tr[k[0]],Tg[k[1]],Tb[k[2]] 

def getab(path,n):
  img=im.open(path)
  lenght=len(np.array(img))#32
  a=np.zeros(lenght*lenght)
  b=np.zeros(lenght*lenght)
  for i in range(0,lenght):
      for j in range(0,lenght):
        index=n*i+j
        a[index]=(i-lenght/2)/lenght
        b[index]=(j-lenght/2)/lenght
  return a,b
  
def create(sub,n):
  path1="/content/subimages/{}.png".format(sub)
  path2="/content/subimages/{}.png".format(sub+1)
  path3="/content/subimages/{}.png".format(sub+n)
  path4="/content/subimages/{}.png".format(sub+1+n)
  a,b=getab(path1,n)
  lenght=len(np.array(im.open(path1)))
  newr=np.zeros(lenght*lenght)
  newg=np.zeros(lenght*lenght)
  newb=np.zeros(lenght*lenght)
  for i in range(0,lenght):
    for j in range(0,lenght):
      index=lenght*i+j
      print(index,"pixel under processing")
      t1,t2,t3,t4=list([getT(path1,path1,index)][0]),list([getT(path2,path1,index)][0]),list([getT(path3,path1,index)][0]),list([getT(path4,path1,index)][0])
      newr[index]=(1-a[index])*((1-b[index])*t1[0]+b[index]*t2[0])+a[index]*((1-b[index])*t3[0]+b[index]*t4[0])
      newg[index]=(1-a[index])*((1-b[index])*t1[1]+b[index]*t2[1])+a[index]*((1-b[index])*t3[1]+b[index]*t4[1])
      newb[index]=(1-a[index])*((1-b[index])*t1[2]+b[index]*t2[2])+a[index]*((1-b[index])*t3[2]+b[index]*t4[2])
  newr=im.fromarray(newr.reshape((lenght,lenght))).convert('L')
  newg=im.fromarray(newg.reshape((lenght,lenght))).convert('L')
  newb=im.fromarray(newb.reshape((lenght,lenght))).convert('L')
  new_img=im.merge("RGB",[newr,newg,newb])
  plt.imshow(im.open(path1))
  plt.show()
  plt.imshow(new_img)
  plt.show()
  new_img.save(path1)    

def recompose(ori_path,sub_path,n,m):
   result=im.new("RGB",(128,128))
   img_list=[]
   for i in range(0,len(os.listdir(sub_path))):
     img_list.append("{}/{}".format(sub_path,i))
   for i in range(0,n):
     for j in range(0,m):
       img=img_list[i*n+j]
       result.paste(im.open(img+".png"),box=(j*int(128/n),i*int(128/m)))
   plt.imshow(result)
   plt.show()
   result.save("img_recompose.png")

def one(path,n,m):
  image=(im.open(path)).resize((128,128))
  image_list = cut_image(image,n,m)
  save_images(image_list)

def two(path):
  img=im.open(path)
  r,g,b,a=img.split()#"L"

  plt.title('oir_histogram',fontsize='large')
  create_histogram(r,g,b,"oir_histogram.png")
  fr,fg,fb=fitter(r),fitter(g),fitter(b)
  plt.title('new_histogram',fontsize='large')
  create_histogram(fr,fg,fb,"new_histogram.png")
  Pr,Pg,Pb=p(fr),p(fg),p(fb)
  Tr,Tg,Tb=t(Pr),t(Pg),t(Pb)
  newr,newg,newb=recreate(r,Tr),recreate(g,Tg),recreate(b,Tb)
  new_img=im.merge("RGB",[newr,newg,newb])
  
  plt.title('original image',fontsize='large') 
  plt.imshow(img)
  plt.show()
  plt.title('strenghted image',fontsize='large') 
  plt.imshow(new_img)
  plt.savefig("strenght_image.png")
  plt.show()

  plt.title('original r image',fontsize='large')
  plt.imshow(r)
  plt.show()
  plt.title('original g image',fontsize='large')
  plt.imshow(g)
  plt.show()
  plt.title('original b image',fontsize='large')
  plt.imshow(b)
  plt.show()
  plt.title('strenghted r image',fontsize='large')
  plt.imshow(newr)
  plt.show()
  plt.title('strenghted g image',fontsize='large')
  plt.imshow(newg)
  plt.show()
  plt.title('strenghted b image',fontsize='large')
  plt.imshow(newb)
  plt.show()

def third(ori_path,sub_path,n,m):
  
  for i in range(1,n-1):
    for j in range(1,m-1):
         index=i*n+j
         create(index,n)
  recompose(ori_path,sub_path,n,m)

if __name__ == '__main__':
    n=4
    m=4
    ori_img = "/content/original_image.PNG"
    file_path="/content/subimages"
    sub_img="/content/subimages/14.png"
    one(ori_img,n,m)
    two(sub_img)
    third(ori_img,file_path,n,m)
