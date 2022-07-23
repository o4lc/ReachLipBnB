from packages import *
from utilities import fx, Q, plot_space
from BB_Node_Class import BB_node

class Branch_Bound:
    def __init__(self, coord_up=None, coord_low=None, verbose=False, eta=1e-3, dim=2, eps=0.1):
        self.space_nodes = [BB_node(np.infty, -np.infty, coord_up, coord_low)]
        self.BUB = None
        self.BLB = None
        self.init_coord_up = coord_up
        self.init_coord_low = coord_low
        self.verbose = verbose
        self.eta = eta
        self.PGD_iter = 5
        self.bach_number = 1
        self.dim = dim
        self.eps = eps

    def prune(self):
        for node in self.space_nodes:
            if node.lower >= self.BUB:
                self.space_nodes.remove(node)
                if self.verbose:
                    print('deleted')

    def lower_bound(self, index):
        temp = self.space_nodes[index].coord_lower
        return (0.25 * temp.T @ Q @ temp - 1 / (10 + 10 * len(self.space_nodes)))

    def upper_bound(self, index):
        # add gradient descent
        x0 = np.random.uniform(low = self.space_nodes[index].coord_lower, 
                                          high = self.space_nodes[index].coord_upper,
                                            size=(self.bach_number, self.dim))

        x = Variable(torch.from_numpy(x0.astype('float')).float(), requires_grad=True)
        
        for i in range(self.PGD_iter):
            x.requires_grad = True
            for j in range(self.bach_number):
                with torch.autograd.profiler.profile() as prof:
                    ll = fx(x[j], torch)
                    ll.backward()
                    # l.append(ll.data)

            with no_grad():
                gradient = x.grad.data
                x = x - self.eta * gradient

            # Calculating the diagonals together
            # with no_grad():
            #     print(x)
            #     print("+")
            #     gradient = jacobian(expression_reducer, x)
            #     print('grad', gradient)
            #     x = x - self.eta * gradient

        # # x = torch.clip(x, torch.from_numpy(self.space_nodes[index].coord_lower), 
        # #                         torch.from_numpy(self.space_nodes[index].coord_upper)).float()

        # # x = torch.clamp(x, torch.from_numpy(self.space_nodes[index].coord_lower).float(), 
        # #                         torch.from_numpy(self.space_nodes[index].coord_upper).float()).float()

        x = torch.max(torch.min(x, torch.from_numpy(self.space_nodes[index].coord_upper).float()),
                        torch.from_numpy(self.space_nodes[index].coord_lower).float())

        ub = np.min([fx(xx, torch) for xx in x])

        return ub

    def branch(self):
        # Prunning Function
        self.prune()

        # Choosing the node to branch
        max_score, max_index = -1, -1
        for i in range(len(self.space_nodes)):
            if self.space_nodes[i].score > max_score:
                max_index = i
                max_score = self.space_nodes[i].score

        coord_to_split = np.argmax(self.space_nodes[max_index].coord_upper 
                                   - self.space_nodes[max_index].coord_lower)
        
        # This can be optimized by keeping the best previous 'x's in that space
        node = self.space_nodes.pop(max_index)
        node_l = np.array(node.coord_lower, dtype=float)
        node_u = np.array(node.coord_upper, dtype=float)


        new_axis = (node.coord_upper[coord_to_split] + 
                           node.coord_lower[coord_to_split])/2

        node_split_u1 = np.array(node_u, dtype=float)
        node_split_u1[coord_to_split] = new_axis

        node_split_l2 = np.array(node_l, dtype=float)
        node_split_l2[coord_to_split] = new_axis

        self.space_nodes.append(BB_node(np.infty, -np.infty, node_split_u1, node_l))
        self.space_nodes.append(BB_node(np.infty, -np.infty, node_u, node_split_l2))
                
        return [len(self.space_nodes) - 2, len(self.space_nodes) - 1], node.upper, node.lower

    def bound(self, index, parent_ub, parent_lb): 
        cost_upper = self.upper_bound(index)
        cost_lower = self.lower_bound(index)

        self.space_nodes[index].lower = max(cost_lower, parent_lb)
        self.space_nodes[index].upper = cost_upper
        # self.space_nodes[index].upper = min(cost_upper, parent_ub)


    def run(self):
        self.BUB = np.infty
        self.BLB = -np.infty

        # print("#1", self.space_nodes)
        self.bound(0, self.BUB, self.BLB)
        while self.BUB - self.BLB >= self.eps:
            indeces, deleted_ub, deleted_lb = self.branch()
            # print("#2", self.space_nodes)
            for ind in indeces:
                self.bound(ind, deleted_ub, deleted_lb)

            self.BUB = np.min([self.space_nodes[i].upper for i in range(len(self.space_nodes))])
            self.BLB = np.min([self.space_nodes[i].lower for i in range(len(self.space_nodes))])
            
            if self.verbose:
                print(self.BLB , self.BUB)
                plot_space(self.space_nodes, self.init_coord_low, self.init_coord_up)
                print('--------------------')



        return self.BLB, self.BUB, self.space_nodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.space_nodes)):
            string += self.space_nodes[i].__repr__() 
            string += "\n"

        return string
        