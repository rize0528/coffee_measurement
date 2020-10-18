from CoffeeMeasure import *


class RegressionModel(CoffeeMeasureCore):
    def __init__(self):
        super().__init__()
        self.essentials = ['rr', 'rb', 'rg', 'rc']
        self.log.info("Essentials : {}".format(self.essentials))
