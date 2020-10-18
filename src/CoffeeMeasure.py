import logging


class CoffeeMeasureCore:
    def check_essential_params(self):
        fields_not_found = [x for x in self.essentials if x not in self.dataframe.columns]
        #
        if len(fields_not_found) > 0:
            self.log.ERROR('Essential parameters for {0}'.format(self.__name__))

    def __init__(self, log_level=logging.INFO):
        #
        self.base_essentials = ['value','']
        log_format = '[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s'
        logging.basicConfig(level=log_level, format=log_format)
        self.log = logging.getLogger()
