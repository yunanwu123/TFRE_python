from TFRE import TFRE 
import numpy as np


n = 100
p = 400
X = np.random.normal(0,1,size=(n,p))
beta =  np.append([1.5,-1.25,1,-0.75,0.5],np.zeros(p-5))
y = X.dot(beta) + np.random.normal(0,1,n)

obj = TFRE()
obj.fit(X,y,eta_list=np.arange(0.09,0.51,0.03))
obj.coef("1st")[:10]
obj.coef("2nd")[:10]
obj.plot()

newX = np.random.normal(0,1,size=(10,p))
obj.predict(newX,"2nd") 