import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from train import main as train_main
from test import main as test_main
from config import parse_config
from time import sleep


class Analyse():
    styles=["bo-","go-","ro-","co-","mo-","yo-","ko-","wo-","bD--","gD--","rD--"]
    def draw_graph(self,X,Y):
        for i in range(len(X)):    
            plt.plot(X[i],Y[i],self.styles[i],label='epochs='+str(int(i)+1),markersize=10)
        fig1=plt.figure(1)
        axes=plt.subplot(111)
        axes.grid(True)
        plt.legend(loc="lower right")
        plt.ylabel('acc')
        plt.xlabel('batch size')
        plt.show()

def main():
    config = parse_config()
    analyse=Analyse()
    epoch = config.epochs[0]
    X,Y=[],[]
    while epoch<=config.epochs[1]:
        x,y=[],[]
        b_size=config.batch_size[0]
        while b_size<=config.batch_size[1]:
            print(f"epoch:{epoch}  b_size:{b_size}")
            train_main(epoch,b_size,config)
            sleep(2)
            acc=test_main(config)
            x.append(b_size)
            y.append(acc)
            b_size*=2
        X.append(x)
        Y.append(y)
        epoch+=1
    analyse.draw_graph(X,Y)
    
if __name__ == "__main__":
    main()