import warnings
warnings.simplefilter("ignore")

import operator
import itertools

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 57), bool, "IN")
# określamy zbór prymitywów
# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)

# logic operators
# Define a new if-then-else function
#tu sami definiujemy prymityw
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

# solution encoding
# jaka bedzie reprezentacja osobnika, jak będzie enkodowany osobnik

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
# to co podajemy tej fcji to odpowiednik wskaźnika na funkcję w c (gp.PrimitiveTree)

#fit function
# określamy funkcję jakosci, ważny będzie dla nas jak dany osobnik rozpoznaje pocztę
# funkcja przyjmuje jako argument osobnika

def evalSpambase(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    #powyżej osobnik z reprezentacji drzewiastej jest przekształcany na kod python
    #funkcja reprezentuje nasze drzewko
    result = sum(bool(func(*mail)) is bool(target) for mail, target in zip(emails_train, targets_train))
    #czy wartość boolowska zwrócona przez drzewko jest taka jak target
    #powstaje lista wypełniona wartościami true lub folse w zależnosci od tego czy są takie same czy różne
    #ilość poprawnychh i niepoprawnych jest zliczana
    return result,

# GP parameters
#określamy parametry algorytmu
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)#zbiór prymitywów, minimalna głebokość, maksymalna głębokość
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evalSpambase)#podpięcie naszej funkcji oceniajacej

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

import multiprocessing

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

# set population size
pop = toolbox.population(n=80)
hof = tools.HallOfFame(5)#dodatkowa pamięc w której zapisujemy jaki był najlepszy osobnik do tej pory

history = tools.Statistics(lambda ind: ind.fitness.values)#zbieramy do historii statystykę populacji
history.register("max_fit", np.max)#referencje do metod, tutaj moglibyśmy podpiąć swoje funkcje
history.register("mean_fit", np.mean)

#prawdopodobieństwo krzyżowania, mutacj, ilosć iteracji
pop, history = algorithms.eaSimple(pop, toolbox, 0.2, 0.1, 200, history, halloffame=hof)
#jeśleli w max_fit i mean_fit dostajemy taki sam wynik to znaczy że jeden osobnik zdominował całą populację
print("Learned spam detection rule: ", hof[-1])

#jednocześnie:
#ekstrakcja cech - wzięcie pod uwagę pewnego podzbioru cech
#nieliniowa transformacja cech
#zbudowany klasyfikator

plt.figure(figsize=(11, 4))
plots = plt.plot(history.select('max_fit'),'c-', history.select('mean_fit'), 'b-')
plt.legend(plots, ('Max fitness', 'Mean fitness'), frameon=True)
plt.ylabel('Fitness'); plt.xlabel('Iterations'); plt.grid()
#rysujemy proces ewolucyjny