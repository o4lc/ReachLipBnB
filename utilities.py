from fileinput import filename
from packages import *

class Plotter():
    def __init__(self, plotCounter = 0, savingDirectory='./Output/'):
        self.plotCounter = plotCounter
        self.savingDirectory = savingDirectory
        self.images = []
        fileNames = next(os.walk(self.savingDirectory), (None, None, []))[2]
        for fileName in fileNames:
            os.remove(self.savingDirectory + fileName)

    def plotSpace(self, space, l, u):
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rectangle = patches.Rectangle((l[0], l[1]), u[0] - l[0], u[1] - l[1], edgecolor=None, facecolor="grey", linewidth=7, alpha=0.5)
        ax.add_patch(rectangle)

        print(space)
        for node in space:
            rectangle = patches.Rectangle((node.coord_lower[0], node.coord_lower[1]), 
                                        node.coord_upper[0] - node.coord_lower[0], 
                                        node.coord_upper[1] - node.coord_lower[1], 
                                        edgecolor=None, facecolor="green", linewidth=7, alpha=0.3)
            ax.add_patch(rectangle)
            ax.vlines(x=node.coord_lower[0], ymin=node.coord_lower[1], 
                                ymax=node.coord_upper[1], color='r', linestyle='--')
            
            ax.vlines(x=node.coord_upper[0], ymin=node.coord_lower[1], 
                                ymax=node.coord_upper[1], color='r', linestyle='--')
            
            ax.hlines(y=node.coord_lower[1], xmin=node.coord_lower[0], 
                                xmax=node.coord_upper[0], color='r', linestyle='--')
            
            ax.hlines(y=node.coord_upper[1], xmin=node.coord_lower[0], 
                                xmax=node.coord_upper[0], color='r', linestyle='--')
        plt.xlim([l[0] - 2, u[0] + 2])
        plt.ylim([l[1] - 2, u[1] + 2])
        # plt.show()
        # self.images.append([ax])
        plt.savefig(self.savingDirectory + str(self.plotCounter))
        self.plotCounter += 1

    def showAnimation(self):
        # Closing any previous open plots
        for j in range(len(plt.get_fignums())):
            plt.close()

        # Creating the gif with the saved pictures
        fig, ax = plt.subplots()
        num = len(next(os.walk(self.savingDirectory), (None, None, []))[2])
        self.images = []
        for i in range(num):
            image = plt.imread(str(self.savingDirectory + str(i) + '.png'))
            im = ax.imshow(image, animated=True)
            self.images.append([im])

        ani = animation.ArtistAnimation(fig, self.images, interval=500, blit=True,
                                repeat_delay=2000)

        plt.axis('off')
        plt.title('Brand and Bound Sequence')
        plt.show()
       