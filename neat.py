#https://github.com/FaydSpeare/NEAT

class Node(object):
    def __init__(self, layer, num):
        self.layer = layer
        self.num = num        
        self.in_val = 0
        self.out_val = 0
        self.recurrent_in_val = 0

    def is_input(self):
        return isinstance(self, Input)

    def is_hidden(self):
        return isinstance(self, Hidden)

    def is_output(self):
        return isinstance(self, Output)

    def activate(self):
        self.out_val = self.in_val

class Input(Node):
    def __init__(self, num):
        super().__init__(0, num)

    def activate(self):
        super().activate()

    def replicate(self):
        node = Input(self.num)
        return node

import math
class Output(Node):
    def __init__(self, num):
        super().__init__(1, num)

    def activate(self):
        self.in_val += self.recurrent_in_val
        try:
            ans = math.exp(-4.9*self.in_val)
        except OverflowError:
            ans = float('inf')
        self.out_val = 1 / (1 + ans)
        self.recurrent_in_val = 0

    def replicate(self):
        node = Output(self.num)
        node.layer = self.layer
        return node

class Hidden(Node):
    def __init__(self, num):
        super().__init__(None, num)

    def activate(self):
        self.in_val += self.recurrent_in_val
        try:
            ans = math.exp(-4.9*self.in_val)
        except OverflowError:
            ans = float('inf')
        self.out_val = 1 / (1 + ans)
        self.recurrent_in_val = 0
    
    def replicate(self):
        node = Hidden(self.num)
        node.layer = self.layer
        return node

class Bias(Node):
    def __init__(self, num):
        super().__init__(0, num)
        self.in_val = 1

    def activate(self):
        super().activate()

    def replicate(self):
        node = Bias(self.num)
        return node

import random
class Connection(object):
    WEIGHT_UB = 3
    WEIGHT_LB = -3
    WEIGHT_STEP = 0.01
    WEIGHT_DIST = ('uniform', 0, 1)
    RANDOM_WEIGHT = 0.1

    def __init__(self, input, output, num):
        self.input = input
        self.output = output        
        self.recurrent = False
        if self.input == self.output:
            self.recurrent = True
        self.weight = self.rand()
        self.enabled = True
        self.num = num

    def feed(self):
        if self.enabled:
            if self.recurrent:
                self.output.recurrent_in_val += self.input.out_val * self.weight
            else:
                self.output.in_val += self.input.out_val * self.weight

    def mutate_weight(self):
        if random.random() < Connection.RANDOM_WEIGHT:
            self.weight = self.rand()
        else:
            self.weight += self.rand() * Connection.WEIGHT_STEP        
        if self.weight > Connection.WEIGHT_UB: self.weight = Connection.WEIGHT_UB
        elif self.weight < Connection.WEIGHT_LB: self.weight = Connection.WEIGHT_LB

    def rand(self):
        if Connection.WEIGHT_DIST[0] == 'uniform':
            width = (Connection.WEIGHT_DIST[2] - Connection.WEIGHT_DIST[1])
            return (random.random() * width) + Connection.WEIGHT_DIST[1]
        elif Connection.WEIGHT_DIST[0] == 'gaussian':
            return random.gauss(Connection.WEIGHT_DIST[1], Connection.WEIGHT_DIST[2])

class Network(object):
    W_MUT = 0.8
    C_MUT = 0.1
    N_MUT = 0.01
    RECURRENT = False
    def next_node_innov(self):
        self.node_innov += 1
        return self.node_innov-1

    def next_conn_innov(self):
        self.conn_innov += 1
        return self.conn_innov-1

    def __init__(self, inputs, outputs, fill=True):
        self.node_innov = 0
        self.conn_innov = 0
        self.hidden_layers = 0
        self.node_innovations = set()
        # list of input nodes
        self.inputs = []
        # dict key = layer, value = list of nodes in layer
        self.hiddens = {}
        # list of output nodes
        self.outputs = []
        # bias node
        self.bias = Bias(self.next_node_innov())
        # list of all nodes w/o bias
        self.nodes = []
        # dict key = input node, value = list of connections
        self.node_conns = {}
        # list of connections w/o bias conns
        self.connections = []
        # list of bias connections
        self.bias_connections = []
        if fill:
            for i in range(inputs):
                node = Input(self.next_node_innov())
                self.inputs.append(node)
                self.nodes.append(node)
            for i in range(outputs):
                node = Output(self.next_node_innov())
                self.outputs.append(node)
                self.nodes.append(node)
            for node in self.inputs:
                self.node_conns[node] = []
            for i in range(outputs):
                connection = Connection(self.bias, self.outputs[i], self.next_conn_innov())
                self.bias_connections.append(connection)
            for i in range(inputs):
                for j in range(outputs):
                    connection = Connection(self.inputs[i], self.outputs[j], self.next_conn_innov())
                    self.node_conns[self.inputs[i]].append(connection)
                    self.connections.append(connection)
            for n in self.nodes:
                self.node_innovations.add(n.num)
            
    def feed_forward(self, inputs):
        results = []
        self.bias.in_val = 1
        self.bias.activate()
        for conn in self.bias_connections:
            conn.feed()
        for i in range(len(self.inputs)):
            node = self.inputs[i]
            node.in_val = inputs[i]
            node.activate()
            for conn in self.node_conns[node]:
                conn.feed()
        for i in range(self.hidden_layers):
            for node in self.hiddens[i+1]:
                node.activate()
                for conn in self.node_conns[node]:
                    conn.feed()
        for node in self.outputs:
            node.activate()
            results.append(node.out_val)
        for node in self.nodes:
            node.out_val = 0
            node.in_val = 0
        return results

    def add_node(self):
        # choose a random connection (not bias connection)
        conn = random.choice(self.connections)
        while node_innovation(conn.input.num, conn.output.num) in self.node_innovations:
            conn = random.choice(self.connections)
        # disable connection
        conn.enabled = False
        # create new node
        node = Hidden(node_innovation(conn.input.num, conn.output.num))
        self.nodes.append(node)
        self.node_innovations.add(node.num)
        # give new node bias connection
        b = Connection(self.bias, node, conn_innovation(self.bias.num, node.num))
        b.weight = 0
        self.bias_connections.append(b)
        # create two new connections
        c1 = Connection(conn.input, node, conn_innovation(conn.input.num, node.num))
        c2 = Connection(node, conn.output, conn_innovation(node.num, conn.output.num))
        self.connections.append(c1)
        self.connections.append(c2)
        self.node_conns[conn.input].append(c1)
        self.node_conns[node] = [c2]
        # fix weights
        c1.weight = 1
        c2.weight = conn.weight
        node.layer = conn.input.layer + 1
        if node.layer == conn.output.layer: # create new layer            
            if node.layer not in self.hiddens: # create if layer n'existe pas
                self.hiddens[node.layer] = [node]                
            else:
                #print("entered at ", node.layer)
                for i in range(node.layer, self.hidden_layers + 1)[::-1]:
                    self.hiddens[i+1] = self.hiddens.pop(i)
                    # increment layers
                    for n in self.hiddens[i+1]:
                        n.layer += 1
                # add new layer
                self.hiddens[node.layer] = [node]
            # increment output layer
            for n in self.outputs:
                n.layer += 1                
            self.hidden_layers += 1                
        else:
            if self.hiddens.get(node.layer) == None:
                self.hiddens[node.layer] = [node]
            self.hiddens[node.layer].append(node)
        return

    def add_connection(self):
        attempts = 0 
        input_node = random.choice(self.nodes)
        output_node = random.choice(self.nodes)
        while ((input_node.layer == output_node.layer) or(self.is_connection(input_node, output_node)))and attempts < 50:
            input_node = random.choice(self.nodes)
            output_node = random.choice(self.nodes)
            attempts += 1 
        if attempts == 50:
            #print("failed attempt=50")
            return
        if input_node.layer > output_node.layer:
            temp = output_node
            output_node = input_node
            input_node = temp
        conn = Connection(input_node, output_node, conn_innovation(input_node.num, output_node.num))
        self.connections.append(conn)
        self.node_conns[input_node].append(conn)
        #print("success attempt=", attempts)

    def is_connection(self, n1, n2):
        if  n1 in self.node_conns:
            for c in self.node_conns[n1]:
                if c.output == n2:
                    return True
        if n2 in self.node_conns:
            for c in self.node_conns[n2]:
                if c.output == n1:
                    return True
        return False

    def randomise_weights(self):
        for conn in self.connections:
            conn.weight = conn.rand()
        for conn in self.bias_connections:
            conn.weight = conn.rand()

    def mutate(self):
        if random.random() < Network.W_MUT:
            for c in self.connections:
                c.mutate_weight()
            for c in self.bias_connections:
                c.mutate_weight()
        if random.random() < Network.C_MUT:
            self.add_connection()
        if random.random() < Network.N_MUT:
            self.add_node()            
        
    def replicate(self):
        clone = Network(len(self.inputs), len(self.outputs), fill=False)
        clone.hidden_layers = self.hidden_layers
        clone.bias = self.bias.replicate()
        clone.node_innov = self.node_innov
        clone.conn_innov = self.conn_innov
        temp_node_map = {}
        for node in self.inputs:
            rep = node.replicate()
            clone.inputs.append(rep)
            clone.nodes.append(rep)
            temp_node_map[rep.num] = rep
        for node in self.outputs:
            rep = node.replicate()
            clone.outputs.append(rep)
            clone.nodes.append(rep)
            temp_node_map[rep.num] = rep
        for node in clone.inputs:
                clone.node_conns[node] = []
        for key in self.hiddens:
            node_list = []
            for node in self.hiddens[key]:
                rep = node.replicate()
                clone.node_conns[rep] = []
                temp_node_map[rep.num] = rep
                node_list.append(rep)
                clone.nodes.append(rep)
            clone.hiddens[key] = node_list
        for conn in self.connections:
            in_node = temp_node_map[conn.input.num]
            out_node = temp_node_map[conn.output.num]
            new_conn = Connection(in_node, out_node, conn.num)
            new_conn.weight = conn.weight
            new_conn.enabled = conn.enabled
            clone.node_conns[in_node].append(new_conn)
            clone.connections.append(new_conn)
        for conn in self.bias_connections:
            out_node = temp_node_map[conn.output.num]
            new_conn = Connection(clone.bias, out_node, conn.num)
            new_conn.weight = conn.weight
            clone.bias_connections.append(new_conn)
        for n in clone.nodes:
                clone.node_innovations.add(n.num)            
        return clone
                    
    def __repr__(self):
        s = "bias:\n"
        s += "  node " + str(self.bias.num) + "\n"
        for c in self.bias_connections:
            s += "    conn " + str(c.num).center(2) + ": " + str(c.input.num).center(2) + " -> " + str(c.output.num).center(2) + " enabled=" + str(c.enabled) + " weight=" +str(c.weight)+"\n"
        s += "inputs:\n"
        for n in self.inputs:
            s += "  node " + str(n.num) + "\n"
            for c in self.node_conns[n]:
                s += "    conn " + str(c.num).center(2) + ": " + str(c.input.num).center(2) + " -> " + str(c.output.num).center(2) + " enabled=" + str(c.enabled) + " weight=" +str(c.weight)+"\n"
        for layer in range(1, self.hidden_layers+1):
            s += "layer " + str(layer) + ":\n"
            for n in self.hiddens[layer]:
                s += "  node " + str(n.num) + "\n"
                for c in self.node_conns[n]:
                    s += "    conn " + str(c.num).center(2) + ": " + str(c.input.num).center(2) + " -> " + str(c.output.num).center(2) + " enabled=" + str(c.enabled) + " weight=" +str(c.weight) + "\n"
        s += "outputs:\n"
        for n in self.outputs:
            s += "  node " + str(n.num) + "\n"
        return s

node_innovations = -1
conn_innovations = -1
node_innov_dict = {}
conn_innov_dict = {}
setup = False
def setup_innovations(io):
    global node_innovations
    global conn_innovations
    global setup
    global node_innov_dict
    global conn_innov_dict    
    node_innovations = (io[0] + io[1] + 1) - 1
    conn_innovations = ((io[0] + 1) * io[1]) - 1
    node_innov_dict = {}
    conn_innov_dict = {}
    setup = True
def node_innovation(before, after):
    global node_innovations
    global conn_innovations
    global setup
    if not setup: raise Exception()
    #print("Asking for", before, "to", after)        
    tup = (before, after)
    if node_innov_dict.get(tup) != None:
        #print("gave", node_innov_dict[tup])
        return node_innov_dict[tup]
    else:
        node_innovations += 1
        node_innov_dict[tup] = node_innovations
        #print("gave", node_innovations)
        return node_innovations
def conn_innovation(start, end):
    global node_innovations
    global conn_innovations
    global setup
    if not setup: raise Exception()    
    tup = (start, end)
    if conn_innov_dict.get(tup) != None:
        return conn_innov_dict[tup]
    else:
        conn_innovations += 1
        conn_innov_dict[tup] = conn_innovations
        return conn_innovations

class Population(object):
    ELITE = 2
    STALE_SPEC = 15
    STALE_POP = 20

    def __init__(self, io, entity, size):
        self.io = io
        self.size = size
        self.entity = entity
        self.population = []
        self.species = []
        self.spec_innov = 0
        self.gen = 0
        self.best_fitness = 0
        self.gen_fitness = 0
        self.best_entity = None
        self.stale = 0
        for i in range(size):
            entity = self.entity(io)
            entity.mutate()
            self.population.append(entity)

    def natural_selection(self):
        #random.seed(a=None)                    
        self.speciate()
        self.calculate_fitness()
        self.sort_species()        
        self.cull_species()
        self.share_fitness()
        # kill stale and bad species
        for spec in self.species:
            if spec.stale > Population.STALE_SPEC and (self.species.index(spec) + 1) >= Population.ELITE:
                self.species.remove(spec)
        fitness_sum = self.get_average_sum()
        for spec in self.species:
            if math.floor((spec.get_average_fitness() / fitness_sum) * self.size) < 1:
                self.species.remove(spec)
        if len(self.species) == 0:
            print("everyone died")
            return True
        children = []
        if self.stale > Population.STALE_POP and len(self.species) > 1:
            no_of_children = math.floor(self.size/2)-1
            for i in range(no_of_children):
                    children.append(self.species[0].progeny())
                    children.append(self.species[1].progeny())
            children.append(self.species[0].champion.replicate())
            children.append(self.species[1].champion.replicate())            
        else:
            for spec in self.species:                
                no_of_children = math.floor((spec.get_average_fitness() / fitness_sum) * self.size)
                if no_of_children >= 5:
                    children.append(spec.champion.replicate())
                    no_of_children -= 1
                for i in range(no_of_children):
                    children.append(spec.progeny())
        self.population = children
        self.gen += 1
        return False
        
    def speciate(self):
        for spec in self.population:
            spec.entities = []
        for entity in self.population:
            suitable_species = False            
            for spec in self.species:
                if Species.are_compatible(spec.standard, entity.brain):
                    spec.add(entity)
                    suitable_species = True
                    break
            if not suitable_species:
                self.species.append(Species(entity, self.spec_innov))
                self.spec_innov += 1

    def sort_species(self):
        for spec in self.species:
            spec.sort()
        self.species.sort(key=lambda x: x.gen_fitness, reverse = True)
        self.gen_fitness = 0
        self.stale += 1
        for spec in self.species:
            if spec.gen_fitness > self.best_fitness:
                self.best_fitness = spec.gen_fitness
                self.stale = 0
                self.best_entity = spec.entities[0].replicate()
            if spec.gen_fitness > self.gen_fitness:
                self.gen_fitness = spec.gen_fitness

    def calculate_fitness(self):
        for spec in self.species:
            spec.calculate_fitness()

    def cull_species(self):
        for spec in self.species:
            spec.cull()

    def share_fitness(self):
        for spec in self.species:
            spec.share_fitness()

    def get_average_sum(self):
        total = 0
        for spec in self.species:
            total += spec.get_average_fitness()
        return total

    def __repr__(self):
        string = ""
        string += "Total Pop: {}\n".format(len(self.population))
        for spec in self.species:
            string += repr(spec)
        return string

    def __getitem__(self, item):
        return self.population[item]

class Species:
    THRESHOLD = 4.0
    ED_COEFF = 1
    W_COEFF = 3.0
    DUP_PARENT = 0.25
    PARENT_WEIGHT = 0.5
    GENE_ENABLE = 0.75

    def __init__(self, first, innov):
        self.entities = []        
        self.standard = first.brain.replicate()
        self.champion = first.replicate()
        self.innov = innov
        self.best_fitness = 0
        self.stale = 0
        self.gen_fitness = 0
        self.entities.append(first)

    def add(self, entity):
        self.entities.append(entity)

    def sort(self):
        self.entities.sort(key=lambda x: x.fitness, reverse = True)
        self.gen_fitness = self.entities[0].fitness
        if self.entities[0].fitness > self.best_fitness:
            self.best_fitness = self.entities[0].fitness
            self.champion = self.entities[0].replicate()
            self.stale = 0
        else:
            self.stale += 1
        self.standard = self.entities[0].brain.replicate()

    def calculate_fitness(self):
        for e in self.entities:
            e.calc_fitness()

    def cull(self):
        if len(self.entities) > 2:
            for i in range(math.floor(len(self.entities)/2)):
                del self.entities[-1]

    def share_fitness(self):
        for e in self.entities:
            e.shared_fitness = e.fitness / len(self.entities)

    def get_average_fitness(self):
        total = 0
        for e in self.entities:
            total += e.shared_fitness
        return total / len(self.entities)

    def select_parent(self):
        fitness_sum = 0
        for e in self.entities:
            fitness_sum += e.fitness
        rand = random.randint(0, math.floor(fitness_sum))
        running_sum = 0
        for e in self.entities:
            running_sum += e.fitness
            if running_sum > rand:
                return e
        return self.entities[0]

    def progeny(self):
        progeny = None
        if random.random() < Species.DUP_PARENT:
            progeny = self.select_parent().replicate()
        else:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            if parent1.fitness < parent2.fitness:
                brain = Species.crossover(parent2.brain, parent1.brain)
                progeny = parent2.child(brain)
            else:
                brain = Species.crossover(parent1.brain, parent2.brain)
                progeny = parent1.child(brain)
        progeny.mutate()
        return progeny

    @staticmethod
    def are_compatible(net1, net2):
        e_and_d = Species.excess_and_disjoint(net1, net2)
        w_diff = Species.weight_diff(net1, net2)
        length = len(net1.connections)+len(net1.bias_connections) + len(net2.connections)+len(net2.bias_connections)        
        N = 1
        comp = ((Species.ED_COEFF * e_and_d) / N) + (Species.W_COEFF * w_diff)
        return comp < Species.THRESHOLD

    @staticmethod
    def crossover(strong_parent, weak_parent):
        child = strong_parent.replicate()
        for conn in child.connections:
            for weak_conn in weak_parent.connections:
                if conn.num == weak_conn.num:
                    if random.random() < Species.PARENT_WEIGHT:
                        conn.weight = weak_conn.weight
                    if not conn.enabled or not weak_conn.enabled:
                        if random.random() < Species.GENE_ENABLE:
                            conn.enabled = False
                        else:
                            conn.enabled = True 
                    break
        for conn in child.bias_connections:
            for weak_conn in weak_parent.bias_connections:
                if conn.num == weak_conn.num:
                    if random.random() < Species.PARENT_WEIGHT:
                        conn.weight = weak_conn.weight
                    break
        return child    

    @staticmethod
    def excess_and_disjoint(net1, net2):
        matched = 0
        for conn in net1.connections:
            for other in net2.connections:
                if other.num == conn.num:
                    matched += 1
                    break
        return len(net1.connections) + len(net2.connections) - 2*matched  

    @staticmethod
    def weight_diff(net1, net2):
        matched = 0
        total = 0
        for conn in net1.connections:
            for other in net2.connections:
                if other.num == conn.num:
                    matched += 1
                    total += abs(conn.weight - other.weight)
                    break
        if matched == 0:
            return 100
        return total/matched

    def __repr__(self):
        string = ""
        string += "Species " + str(self.innov) + ": \n"
        for e in self.entities:
            string += "    "
            string += repr(e) + "\n"
        return string

class Neat(object):
    __isfrozen = False
    def __init__(self, io, entity, stop_condition, size, config=None, verbose=True):
        setup_innovations(io)        
        self.__entity = entity
        self.__io = io
        self.__config = config
        self.__stop_condition = stop_condition
        self.__solved = False
        self.__solvers = []
        self.__verbose = verbose
        self.__pop = Population(io, self.__entity, size)
        self.__setup()
        self.__check_entity()
        self.__isfrozen = True
        
    @property
    def population(self):
        return self.__pop

    @property
    def solvers(self):
        return self.__solvers

    @property
    def stop_condition(self):
        return self.__stop_condition

    @stop_condition.setter
    def stop_condition(self, function):
        self.__stop_condition = function

    def is_solved(self):
        if self.__stop_condition == None:
            if self.__verbose:
                print("Warning: NEAT Object without stop condition cannot be solved")
        return self.__solved

    def next(self):
        #Runs the next iteration of natural selection.
        #If verbose was set to True, information about the generation is displayed     
        self.__pop.natural_selection()
        if self.__verbose:
            s  = "| Gen : {} ".format(str(self.__pop.gen).ljust(3))
            s += "| No. Species : {} ".format(str(len(self.__pop.species)).ljust(3))
            s += "| Score : {} ".format("{:.2f}".format(self.__pop.gen_fitness).ljust(8))
            s += "| HighScore : {} ".format("{:.2f}".format(self.__pop.best_fitness).ljust(8))
            print(s)

    def run(self, iterations = 'inf'):
        #Runs the specified number of iterations of natural selections through next().
        #Parameters: iterations (int): number of iterations (default = inf)
        if iterations == 'inf':
            if self.__stop_condition == None:
                raise Exception("stop condition required for run")
            while not self.__solved:
                self.next()
                if self.__stop_condition != None:
                    for spec in self.__pop.species:
                        e = spec.entities[0]
                        result = self.__stop_condition(e)
                        if result:
                            self.__solvers.append(e)
                        self.__solved = self.__solved or result
            print("SOLVED...")
            print("network=", self.__solvers)
        else:
            for i in range(iterations):
                self.next()
   
    def __check_entity(self):
        if not issubclass(self.__entity, EntityXOR):
            raise Exception("entity must be a subclass of Entity")
        fitness_func = getattr(self.__entity, "calc_fitness", None)
        if not callable(fitness_func):
            raise Exception("entity must implement 'calc_fitness' function")
    
    def __setup(self):
        self.default_config = {    
                # MUTATION RATES
                'weight_mut' : 0.95,
                'connection_mut' : 0.1,
                'node_mut' : 0.05,
                'random_weight' : 0.1,                
                # SPECIES DIFFERENTIATION
                'threshold' : 3.0,
                'disjoint' : 1.0,
                'weights' : 0.5,
                # CROSSOVER
                'dup-parent' : 0.25,
                'weak-parent-weight' : 0.5,
                'gene-enable' : 0.75,
                # NATURAL SELECTION
                'elite' : 2,
                'stale_species' : 15,
                'stale__pop' : 20,
                # WEIGHTS
                'weight_upper_bound' : 3,
                'weight_lower_bound' : -3,
                'weight_step' : 0.01,
                'weight_distr' : ('gaussian', 0, 1),
                # RECURRENT NETWORK
                'recurrent': False
        }
        self.__configure(self.default_config)
        if self.__config != None: self.__configure(self.__config)
        
    def __configure(self, config):       
        if 'weight_mut' in config: Network.W_MUT = config['weight_mut']
        if 'connection_mut' in config: Network.C_MUT = config['connection_mut']
        if 'node_mut' in config: Network.N_MUT = config['node_mut']
        if 'random_weight' in config: Connection.W_MUT = config['random_weight']
        if 'weight_step' in config: Connection.WEIGHT_STEP = config['weight_step']        
        if 'threshold' in config: Species.THRESHOLD = config['threshold']
        if 'disjoint' in config: Species.ED_COEFF = config['disjoint']
        if 'weights' in config: Species.W_COEFF = config['weights']
        if 'dup_parent' in config: Species.DUP_PARENT = config['dup_parent']
        if 'weak-parent_weight' in config: Species.PARENT_WEIGHT = config['weak-parent_weight']
        if 'gene_enable' in config: Species.GENE_ENABLE = config['gene_enable']
        if 'elite' in config: Population.ELITE = config['elite']
        if 'stale_species' in config: Population.STALE_SPEC = config['stale_species']
        if 'stale__pop' in config: Population.STALE__pop = config['stale__pop']
        if 'weight_upper_bound' in config: Connection.WEIGHT_UB = config['weight_upper_bound']
        if 'weight_lower_bound' in config: Connection.WEIGHT_LB = config['weight_lower_bound']
        if 'weight_distr' in config: Connection.WEIGHT_DIST = config['weight_distr']
        if 'recurrent' in config: Network.RECURRENT = config['recurrent']
    
    def __repr__(self):
        return repr(self.__pop)

    def __call__(self):
        self.run()

    def __getattr__(self, name):
        print("Atrribute {} does not exist".format(name))

    def __setattr__(self, name, val):
        if self.__isfrozen and not hasattr(self, name):
            print("Atrribute {} does not exist".format(name))
        object.__setattr__(self, name, val)

class EntityXOR(object):
    def __init__(self, io, net=None):
        self.io = io
        if net == None:
            self.brain = Network(io[0], io[1])
        else:
            self.brain = net
        self.fitness = 0
        self.shared_fitness = 0
        self.solved = False

    def think(self, vision):
        return self.brain.feed_forward(vision)

    def mutate(self):
        self.brain.mutate()

    def replicate(self):
        return self.__class__(self.io, self.brain.replicate())

    def child(self, net):
        return self.__class__(self.io, net)

    def __repr__(self):
        return "Entity: {{ Fitness={:.2f} , Hidden_Nodes={} , Network={} }}".format(float(self.fitness), len(self.brain.hiddens), self.brain)

    def calc_fitness(self):  #interface
        error = 0
        error += (self.think([0,0])[0])**2
        error += (1 - self.think([0,1])[0])**2
        error += (1 - self.think([1,0])[0])**2
        error += (self.think([1,1])[0])**2
        score = 4 - error
        self.fitness = score**2

    def xor_assessment(e):  #function assessment 
        correct = 0
        if e.think([0,0])[0] < 0.5: correct += 1
        if e.think([0,1])[0] > 0.5: correct += 1
        if e.think([1,0])[0] > 0.5: correct += 1
        if e.think([1,1])[0] < 0.5: correct += 1
        if correct == 4:
            return True
        return False

def tpj():
    config = {        
        # MUTATION RATES
        'weight_mut' : 0.95,
        'connection_mut' : 0.1,
        'node_mut' : 0.01,
        'random_weight' : 0.1,        
        # SPECIES DIFFERENTIATION
        'threshold' : 4.0,
        'disjoint' : 1.0,
        'weights' : 3.0,
        # CROSSOVER
        'dup-parent' : 0.25,
        'weak-parent-weight' : 0.5,
        'gene-enable' : 0.75,        
        # NATURAL SELECTION
        'elite' : 2,
        'stale_species' : 15,
        'stale_pop': 20,
        # WEIGHTS
        'weight_upper_bound' : 3,
        'weight_lower_bound' : -3,
        'weight_step' : 0.03,
        'weight_distr' : ('gaussian', 0, 1),
        # RECURRENT NETWORK
        'recurrent' : False
    }   
    neat = Neat(io=(2,1), entity=EntityXOR, stop_condition=EntityXOR.xor_assessment, size=100, config=config)
    neat.run()

if __name__ == '__main__':    
    tpj()
