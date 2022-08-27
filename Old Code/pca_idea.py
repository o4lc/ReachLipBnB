import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    dim = 2
    mean = [10, -10]
    cov = [[1, 3], [3, 10]]
    x, y = np.random.multivariate_normal(mean, cov, 5000).T


    plt.figure()

    plt.subplot(1, 3, 1)
    plt.scatter(x, y)
    plt.axis('square')

    data = np.array([x, y]).T

    pca = PCA()
    data_new = pca.fit_transform(data)

    data_mean = pca.mean_
    data_comp = pca.components_
 

    data -= data_mean
    data_test = data_comp @ data.T
    
    plt.subplot(1, 3, 2)
    plt.scatter(data_new[:, 0], data_new[:, 1])
    plt.axis('square')
    
    plt.subplot(1, 3, 3)
    plt.scatter(data_test[0], data_test[1])
    plt.axis('square')
    
    plt.show()

if __name__ == '__main__':
    main()