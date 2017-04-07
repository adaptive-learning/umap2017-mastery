from abc import ABC, abstractmethod


class MasteryDetection(ABC):

    def __init__(self, data):
        self.detected_mastery = [self.detect_mastery_in_sequence(data.answer[s, :]) for s in range(data.students)]

    @abstractmethod
    def detect_mastery_in_sequence(self, seq):
        pass


class NCCMastery(MasteryDetection):

    def __init__(self, data, n):
        self.n = n
        MasteryDetection.__init__(self, data)

    def detect_mastery_in_sequence(self, seq):
        in_row = 0
        for i in range(len(seq)):
            if seq[i]:
                in_row += 1
            else:
                in_row = 0
            if in_row == self.n:
                return i
        return len(seq) 


class EMAMastery(MasteryDetection):

    def __init__(self, data, alpha, threshold):
        self.alpha = alpha
        self.threshold = threshold
        MasteryDetection.__init__(self, data)

    def detect_mastery_in_sequence(self, seq):
        skill = 0
        for i in range(len(seq)):
            skill = self.alpha * skill + (1-self.alpha) * seq[i]
            if skill > self.threshold:
                return i
        return len(seq) 
        
        
class BKTMastery(MasteryDetection):

    def __init__(self, data, params, threshold):
        self.params = params
        self.threshold = threshold
        MasteryDetection.__init__(self, data)

    def detect_mastery_in_sequence(self, seq):
        skill = self.params['init']
        slip, guess = self.params['slip'], self.params['guess'] 
        for i in range(len(seq)):
            if seq[i]:
                P = skill * (1-slip) / (skill * (1-slip) + (1-skill) * guess)
            else:
                P = skill * slip / (skill * slip + (1-skill) * (1 - guess))
            skill = P + (1-P) * self.params['learn']            
            if skill > self.threshold:
                return i
        return len(seq) 
