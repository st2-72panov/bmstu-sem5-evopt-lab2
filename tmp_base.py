class GeneticAlgorithm():
    @dataclass
    class Constants():
        minimization: bool
        dims: int
        A: np.ndarray  # list of left borders (for each coordinate)
        B: np.ndarray  # list of right borders (for each coordinate)
        dX: np.ndarray
        mutation_chance: float
        N: int
        threshold: float
        max_generations: int = 1000

    @dataclass
    class PlotData():
        mean_x_history: list = field(default_factory=list)
        mean_y_history: list = field(default_factory=list)
        mean_fitness_history: list = field(default_factory=list)
        std_x_history: list = field(default_factory=list)
        std_y_history: list = field(default_factory=list)
        std_fitness_history: list = field(default_factory=list)
        
        best_y_history: list = field(default_factory=list)

    def __init__(self, f: Callable, constants: Constants):
        self.f0 = f
        self.f = f
        if constants.minimization:
            self.f = lambda X: -f(X)
        self.constants = constants
        self.constants.N = (self.constants.N + self.constants.N % 2) // 2 * 2
        
        # calculate gene_length & adjust dX
        m_vals = (self.constants.B - self.constants.A) // self.constants.dX
        self.gene_length = np.ceil(np.log2(m_vals + 1)).astype(int)                # np.ceil, np.log2
        self.constants.dX = (self.constants.B - self.constants.A) / (2 ** self.gene_length - 1)
        
        # initialize population
        population = []
        m_vals = ((self.constants.B - self.constants.A) // self.constants.dX).astype(int)
        for d in range(self.constants.dims):
            traits_10 = np.random.randint(0, m_vals[d] + 1, size=self.constants.N)  # np.random.randint
            traits_2 = [bin(val)[2:].zfill(self.gene_length[d]) for val in traits_10]
            population.append(traits_2)
        
        self.entities = np.array(population).T
        self.generation = 1
        self.plot_data = self.PlotData()

    def convert_to_X(self, entity: np.ndarray) -> np.ndarray:
        return self.constants.A + self.constants.dX * np.array([int(trait, 2) for trait in entity])

    def update_metrics(self):
        # renew y_array
        X_list = [self.convert_to_X(entity) for entity in self.entities]
        self.y_array = np.array([self.f(X) for X in X_list])
        self.old_best_entity, self.old_best_y = self.cur_best_entity, self.cur_best_y
        self.cur_best_entity, self.cur_best_y = max(zip(self.entities, self.y_array), key=lambda pair: pair[1])
        if self.cur_best_y > self.best_y:
            self.best_X = self.convert_to_X(self.cur_best_entity)
            self.best_y = self.cur_best_y * (-1 if self.constants.minimization else 1)

        # renew fitness_array    
        y_min = np.min(self.y_array)
        self.fitness_array = np.power(self.y_array - y_min, 4) + self.constants.threshold
        
        # renew cumulative_ratios
        s = np.sum(self.fitness_array)                                            # np.sum
        if s == 0:
            ratios = np.full(len(self.fitness_array), 1/len(self.fitness_array))  # np.full
        else:
            ratios = self.fitness_array / s
        self.cumulative_ratios = np.cumsum(ratios)                                       # np.cumsum
        
        # save data for plots
        self.save_data_for_plots()
    
    
    def get_from_roulette(self) -> np.ndarray:
        r = np.random.random()                                       # np.random.random
        selected_index = np.searchsorted(self.cumulative_ratios, r)  # np.searchsorted
        return self.entities[selected_index]

    def mutate(self, entity: np.ndarray) -> None:
        for d in range(self.constants.dims):
            trait_array = np.array(list(entity[d]), dtype='U1')                                  
            mutation_mask = np.random.random(len(trait_array)) < self.constants.mutation_chance  # np.random.random
            trait_array[mutation_mask] = np.where(trait_array[mutation_mask] == '1', '0', '1')   # np.where
            entity[d] = ''.join(trait_array)

    def cross(self, ent1: np.ndarray, ent2: np.ndarray) -> List[np.ndarray]:
        parents = [ent1, ent2]
        children = []
        for _ in range(2):
            child = []
            for d in range(self.constants.dims):
                t = np.random.randint(1, self.gene_length[d] - 2)  # np.random.randint
                parent_indices = np.random.randint(0, 2, size=2)   # np.random.randint
                part1 = parents[parent_indices[0]][d][:t]    # np.random.randint
                part2 = parents[parent_indices[1]][d][t:]
                child.append(part1 + part2)
            children.append(np.array(child))
        return children
    

    def update_generation(self) -> None:
        children = []
        for _ in range(self.constants.N // 2):
            p1 = self.get_from_roulette()
            p2 = self.get_from_roulette()
            children.extend(self.cross(p1, p2))
        for child in children:
            self.mutate(child)
        self.entities = np.array(children)

    def conduct_evolution(self) -> None:
        self.best_y = -np.inf
        self.cur_best_y, self.cur_best_entity = -np.inf, None
        self.update_metrics()

        while True:
            self.update_generation()
            self.generation += 1
            self.update_metrics()
            
            if (0 < abs(self.cur_best_y - self.old_best_y) <= self.constants.threshold
                or self.generation > self.constants.max_generations):
                return

    def save_data_for_plots(self):
        X_list = np.array([self.convert_to_X(entity) for entity in self.entities])
        
        self.plot_data.mean_x_history.append(np.mean(X_list))
        self.plot_data.mean_y_history.append(np.mean(self.y_array))
        self.plot_data.mean_fitness_history.append(np.mean(self.fitness_array))
        self.plot_data.std_x_history.append(np.std(X_list))
        self.plot_data.std_y_history.append(np.std(self.y_array))
        self.plot_data.std_fitness_history.append(np.std(self.fitness_array))
        
        self.plot_data.best_y_history.append(self.cur_best_y)