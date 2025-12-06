import numpy as np

class ART:
    def __init__(self, n, m, rho, gamma=0.5):
        """
        aqui arranca la red. le pasamos los parametros basicos.
        n: tamaño de lo que entra (los pixeles de la imagen).
        m: cuantas categorias maximo queremos que aprenda.
        rho: que tan exigente es la red (0 es barco, 1 es muy estricto).
        gamma: un numero fijo que se usa para las formulas de ajuste.
        """
        self.n = n #neuronas de entrada
        self.m = m  #neuronas de salida
        self.rho = rho  #parametro de vigilancia
        self.gamma = gamma #Estabilizador
        
        # aqui inicializamos la matriz W con la formula estandar: 1 / (1 + n)
        # es para que todas empiecen parejo
        initial_w = 1.0 / (1.0 + self.n)
        self.W = np.full((m, n), initial_w)

        # la matriz U empieza llena de unos
        self.U = np.ones((m, n))
        
        # este arreglo nos dice cuales neuronas ya tienen dueño (categoria asignada)
        # al principio todas estan libres (False)
        self.committed_neurons = np.zeros(m, dtype=bool)
        self.num_committed_neurons = 0

    def train(self, pattern):
        
        # checamos si el patron no viene vacio, si es 0 pues no hacemos nada
        pattern_sum = np.sum(pattern)
        if pattern_sum == 0:
            return -1, False 

        # paso 1: calculamos la salida multiplicando los pesos W por el patron
        f2_outputs = np.dot(self.W, pattern)

        # ordenamos las neuronas de la que dio mas alto a la mas baja
        # para ver quien gana la competencia
        sorted_neurons = np.argsort(f2_outputs)[::-1]

        # revisamos las neuronas ganadoras una por una
        for neuron_index in sorted_neurons:

            # si la neurona esta vacia, nos la saltamos, solo checamos las que ya aprendieron algo
            if not self.committed_neurons[neuron_index]:
                continue

            # paso 2: prueba de vigilancia
            # comparamos lo que sabia la neurona (U) con lo nuevo que llego (pattern)
            match_vector = np.minimum(self.U[neuron_index, :], pattern)
            match_vector_sum = np.sum(match_vector)
            
            # sacamos el porcentaje de coincidencia
            match_degree = match_vector_sum / pattern_sum

            # si el parecido es mayor a lo que exigimos (rho), entonces esa neurona gana y aprende
            if match_degree >= self.rho:
                
                # actualizamos la matriz W con la formula de aprendizaje
                self.W[neuron_index, :] = match_vector / (self.gamma + match_vector_sum)
                
                # actualizamos la matriz U (el prototipo)
                self.U[neuron_index, :] = match_vector
                
                # regresamos cual gano y False porque no es nueva, ya existia
                return neuron_index, False 

        # si ninguna de las que existian paso la prueba, checamos si hay espacio para una nueva
        if self.num_committed_neurons < self.m:
            new_neuron_index = self.num_committed_neurons
            
            # como es nueva, guardamos el patron tal cual en W y U usando las formulas base
            self.W[new_neuron_index, :] = pattern / (self.gamma + pattern_sum)
            
            self.U[new_neuron_index, :] = pattern
            
            # marcamos esa neurona como ocupada y aumentamos el contador
            self.committed_neurons[new_neuron_index] = True
            self.num_committed_neurons += 1
            return new_neuron_index, True # regresamos True porque si fue categoria nueva
        else:
            # si ya no hay espacio ni coincidencia, pues ni modo, regresamos error (-1)
            return -1, False

    def test(self, pattern):
    
        pattern_sum = np.sum(pattern)
        if pattern_sum == 0:
            return -1

        # calculamos salidas y ordenamos igual que arriba
        f2_outputs = np.dot(self.W, pattern)
        sorted_neurons = np.argsort(f2_outputs)[::-1]

        for neuron_index in sorted_neurons:
            # solo buscamos en las ocupadas
            if not self.committed_neurons[neuron_index]:
                continue
            
            # hacemos la prueba de vigilancia
            match_vector = np.minimum(self.U[neuron_index, :], pattern)
            match_degree = np.sum(match_vector) / pattern_sum

            # si pasa la prueba, encontramos al ganador y lo regresamos
            if match_degree >= self.rho:
                return neuron_index 

        # si no se parece a nada de lo que conoce
        return -1 

    def set_vigilance(self, rho):
        # establece el valor del parametro de vigilancia
        self.rho = rho
        
    def reset(self):
        # esto borra todo y regresa la red a su estado incial
        
        # reiniciamos W a su valor inicial
        initial_w = 1.0 / (1.0 + self.n)
        self.W = np.full((self.m, self.n), initial_w)
        
        # reiniciamos U a puros unos
        self.U = np.ones((self.m, self.n))
        # marcamos todas como desocupadas
        self.committed_neurons = np.zeros(self.m, dtype=bool)
        self.num_committed_neurons = 0
        
    def get_committed_prototypes(self):
        # funcion extra para ver que ha aprendido la red hasta el momento
        prototypes = []
        for i in range(self.num_committed_neurons):
            if self.committed_neurons[i]:
                prototypes.append((i, self.U[i, :]))
        return prototypes 