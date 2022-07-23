from packages import *

Q =  np.array([[2, 0], [0, 1]])

def fx(x, pkg=np):
    if pkg==np:
        cost = 0.5 * x.T @ Q @ x
    else:
        cost = 0.5 * pkg.t(x) @ pkg.from_numpy(Q.astype('float')).float() @ x
    return cost

# Needed for Jacobian Calculations
# def expression_reducer(x):
#     cost = torch.diag(0.5 * x @ torch.from_numpy(Q.astype('float')).float() @ torch.transpose(x, 0, 1))
#     return cost


def plot_space(space, l, u):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((l[0], l[1]), u[0], u[1], edgecolor=None, facecolor="grey", linewidth=7, alpha=0.5)
    ax.add_patch(rectangle)

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
    plt.show()