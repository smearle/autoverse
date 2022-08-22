class Variable():
    def __init__(self, name: str, initial_value=0):
        self.name = name
        self._initial_value = initial_value
        self.value = initial_value

    def increment(self):
        print(f'{self.name} increment')
        self.value += 1

    def decrement(self):
        self.value -= 1

    def reset(self):
        self.value = self._initial_value