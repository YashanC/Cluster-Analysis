import numpy as np
import gzip
import cPickle
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.manifold import TSNE

f = gzip.open('mnist.pkl.gz')
training,validation,test = cPickle.load(f)
f.close()

test_x = np.array([np.asarray(x) for x in test[0]],dtype=np.float64)
test_y = np.asarray(test[1])
test_x = test_x.T

covar = np.cov(test_x)
eigenval,eigenvec = np.linalg.eig(covar)
eigens = [(np.abs(eigenval[i]),eigenvec[:,i]) for i in range(len(eigenval))]
eigens.sort(key=lambda x: x[0], reverse=True)
dim = 1
while sum(eigenval[:dim])/sum(eigenval)<0.9 and dim<=len(eigens):
    dim += 1
W = np.hstack([eigens[i][1].reshape(784,1) for i in range(dim)])
print dim
transform = (W.T.dot(test_x)).T
infoloss = [eigenval[i]/sum(eigenval) for i in range(len(eigenval))]
X = np.linspace(1,len(eigenval),len(eigenval))
plt.figure(0)
plt.scatter(X,infoloss)
plt.title('Information Loss')
plt.xlabel('Eigenvalue')
plt.ylabel('Portion of Variance')
result = TSNE(n_components=2,perplexity=100.0).fit_transform(transform)
"""cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = (.5,.5,.5,1.0)
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.linspace(0,10,11)"""
cmap = plt.get_cmap('jet',10)
plt.figure(1)
plt.title('MNIST')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
#norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
plt.scatter(result[:,0],result[:,1],c=test_y,cmap=cmap)
tic = (np.arange(10)+0.5)*0.9
cbar = plt.colorbar(ticks=tic,label='digits')
cbar.set_ticklabels(np.arange(10))
plt.show()
