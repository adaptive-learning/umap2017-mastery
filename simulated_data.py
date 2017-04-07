import math
import numpy as np
import random
import json
from abc import ABC, abstractmethod


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class SimulatedData(ABC):

    def __init__(self, params, students, length=50):
        self.params = params
        self.students = students
        self.length = length
        self.prob = np.empty((students, length)) * np.nan
        self.answer = np.empty((students, length)) * np.nan
        self.true_mastery = np.ones(students) * length
        self.generate_probabilities()
        self.generate_answers()

    @abstractmethod
    def generate_probabilities(self):
        pass

    def generate_answers(self):
        for s in range(self.students):
            for i in range(self.length):
                if random.random() < self.prob[s, i]:
                    self.answer[s, i] = 1
                else:
                    self.answer[s, i] = 0


class SimulatedDataBKT(SimulatedData):
                
    def generate_probabilities(self):
        for s in range(self.students):
            skill = 0
            if random.random() < self.params["init"]:
                skill = 1
                self.true_mastery[s] = 0
            
            for i in range(self.length):
                if skill == 0:
                    self.prob[s, i] = self.params["guess"]
                else:
                    self.prob[s, i] = 1 - self.params["slip"]
                if skill == 0 and (random.random() < self.params["learn"]):
                    skill = 1
                    self.true_mastery[s] = i + 1


class SimulatedDataLogistic(SimulatedData):    
    MASTERY_THRESHOLD = 0.95  # ground truth mastery threshold
                
    def generate_probabilities(self):
        # init_skill = normal(init, init_var)
        init_skill = self.params["init"] * np.ones(self.students) + \
            self.params["init_var"] * np.random.randn(self.students)
        # delta = normal(learn, learn_var)
        delta = self.params["learn"] * np.ones(self.students) + \
            self.params["learn_var"] * np.random.randn(self.students)
        delta = np.maximum(delta, np.zeros(self.students))
        for s in range(self.students):
            for i in range(self.length):
                self.prob[s, i] = sigmoid(init_skill[s] + delta[s] * i)
                if self.true_mastery[s] == self.length and self.prob[s,i] > self.MASTERY_THRESHOLD:
                    self.true_mastery[s] = i


scenarios = json.load(open("scenarios.json"))
model_map = {"BKT": SimulatedDataBKT, "Logistic": SimulatedDataLogistic}


def data_for_scenario(sc_name, students=100, length=50):
    scenario = scenarios[sc_name]
    data = model_map[scenario["type"]](scenario["params"], students, length)
    return data

