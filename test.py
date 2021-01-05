from Kmean_color import Kmeans
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("12.jpeg")
img_size = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
X= img.reshape((img_size.shape[0]*img_size.shape[1],3))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X[:, 0], X[:, 1], 'bo', markersize=10)
plt.plot()
plt.show()
km=Kmeans(n_clusters=5)
km.fit(X)
print(len(km.labels))
print(len(km.predict(X)))
print(km.compute_centroids(X,km.labels))