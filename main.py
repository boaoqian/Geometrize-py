from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    img = Image.open(path)
    img = img.resize((256,256))
    if img.mode != "RGB":
        img = img.convert('RGB')
    return np.array(img).transpose(2,0,1)

#创建椭圆图像
def draw_ellipse(img, pos, color):
    if type(img) == list:
        for i, p, c in zip(img,pos,color):
            c = tuple((c*255).astype(np.int16))
            p = tuple((p*256).astype(np.int16))
            #半透明
            alpha = (128,)
            draw = ImageDraw.Draw(i, "RGBA")
            draw.ellipse(p,c+alpha)
    else:
        c = tuple(color)
        draw = ImageDraw.Draw(img)
        draw.ellipse(pos,c+alpha)

#三角形
def draw_triangle(img, pos, color):
    if type(img) == list:
        for i, p, c in zip(img,pos,color):
            c = tuple((c*255).astype(np.int16))
            p = tuple((p*256).astype(np.int16)) #p(x1,y1,x2,y2,x3,y3)
            #半透明
            alpha = (128,)
            draw = ImageDraw.Draw(i, "RGBA")
            draw.polygon(p, c+alpha)
    else:
        c = tuple(color)
        draw = ImageDraw.Draw(img)
        draw.ellipse(pos,c)

def compare(src, img):
    img = np.array([np.array(i) for i in img]).transpose(0, 3, 1, 2)
    img = img/256
     #余弦相似度
    img = img.reshape(img.shape[0],-1)
    distance = np.matmul(img, src[0]) / (np.linalg.norm(img, axis=1)*src[1]) #src[0]原图像 src[1]原图范数
    return distance
    
def evolve(score, chromosome, batch):
    rank = np.argsort(score)
    chromosome = chromosome[rank[-10:]] #淘汰,保留前十
    next_generation = [chromosome]
    #杂交 生产子代数 （batch - 前10名）/ 2
    for _ in range(int((batch/10-1)//3)):
        crossbreed = chromosome.copy()
        np.random.shuffle(crossbreed)
        crossbreed = (chromosome + crossbreed)/2
        next_generation.append(crossbreed)
    #变异
    for r in range(int(batch/10 - (batch/10-1)//3 - 1)):
        mutation = (np.random.rand(*chromosome.shape)-0.5)*(r+1)*0.1
        next_generation.append(np.clip(chromosome + mutation, 0, 1))

    return np.concatenate(next_generation) 


def fit(src, num, batch=60,mode='triangle'):
    backcolor = tuple(src.mean(axis=(1,2)).astype(np.int16))
    src = src.reshape(-1) #展成向量方便后续计算相似度
    src = (src, np.linalg.norm(src))#提前求解范数，避免重复计算
    #创建预览
    plt.ion()

    for i in range(num):
        #初始化第一代
        if mode == 'ellipse':
            draw = draw_ellipse
            pos = np.random.rand(batch,2)
            pos = np.concatenate((pos, pos+np.random.rand(batch,2)), axis=1)
        elif mode == 'triangle':
            draw = draw_triangle
            pos = np.random.rand(batch, 6)
        else:
            print("unkown shape")
            exit(1)
        pos = np.clip(pos, 0, 1)
        color = np.random.rand(batch,3)
        last_score = 0
        counter = 0 #更新停滞次数
        g = 1#子代数

        #每一张图片的更新
        while(True):
        # for _ in range(30):
            #优化到score变化小于1
            if i == 0:
                #如果是第一代生成根据平均值生成图片背景颜色
                img = [Image.new("RGB", (256,256), backcolor) for i in range(batch)]
            else:
                img = [last_img.copy() for _ in range(batch)]
            draw(img, pos, color)
            score = compare(src, img)
            now_score = np.max(score)
            #若相似度增加少则停止
            if now_score - last_score < 0.0005:
                counter += 1
                if counter >= 3:
                    break
            else:
                counter = 0
            last_score = now_score
            print(f"{i+1} shapes :generation:{g} score:{last_score:.4f}\r")
            pos = evolve(score, pos, batch)
            color = evolve(score, color, batch)
            g+=1
        if g >= 4:
            last_img = img[np.argmax(score)]
        plt.clf()  # 清除之前画的图
        plt.imshow(last_img)
        plt.pause(0.01)  # 暂停一段时间，不然画的太快会卡住显示不出来
    return last_img.copy()


path = "2023-09-13-200734.jpg"
img = load(path)
shape = (1024,1024)
fit(img, 200).resize(shape).show()
plt.ioff()  # 关闭画图窗口
