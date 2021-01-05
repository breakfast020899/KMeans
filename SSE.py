import cv2
import matplotlib.pyplot as plt
from Kmean_color import Kmeans
def figure_SSE(path):
    img = cv2.imread(path)
    #image2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X=img.reshape((image2.shape[0]*image2.shape[1],3))
		#dinh dang hinh anh thanh 1 danh sach pixel ma tran Mxn,voi 3 mau RGB
	#X = image2.reshape((image2.shape[0] * image2.shape[1], 3))
    #img_size =img.cvtColor(image, cv2.COLOR_BGR2RGB)
    #X=img.reshape((img_size[0] * img_size[1],3))
    sse = []
    list_k = list(range(1, 10))

    for k in list_k:
        km = Kmeans(n_clusters=k)
        km.fit(X)
        sse.append(km.error)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance');
    plt.show()
