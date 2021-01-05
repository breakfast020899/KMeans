import matplotlib.pyplot as plt
import cv2
import numpy as np
from Kmean_color import Kmeans
def Figure_colors(path,n_clusters):
    img = cv2.imread(path)
    img_size= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X = img.reshape((img_size.shape[0]*img_size.shape[1],3))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X[:, 0], X[:, 1], 'bo', markersize=10)
    plt.plot()
    plt.show()
    ####################################################
    km = Kmeans(n_clusters=n_clusters)
    km.fit(X)
    centers = []
    plt.xlabel('x') # label trục x
    plt.ylabel('y') # label trục y
    #plt.zlabel('z')
    plt.title("Kmeans") # title của đồ thị
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] # danh sách các màu hỗ trợ
    for i in range(km.n_clusters):
        centroids=km.centroids
        centers.append(centroids)
        #print(centers)
        data = X[km.labels == i] # lấy dữ liệu của cụm i
        plt.plot(data[:, 0], data[:, 1], plt_colors[i]+'o', markersize = 4, label = 'cluster_' + str(i)) # Vẽ cụm i lên đồ thị
        #plt.plot(centers[i][0], centers[i][1], plt_colors[i] + '*', markersize = 4, label = 'center_' + str(i)) # Vẽ tâm cụm i lên đồ thị
    plt.legend() # Hiện bảng chú thích
    plt.show()
