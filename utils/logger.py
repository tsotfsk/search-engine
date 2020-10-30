import logging
import sys


class Logger(object):

    def __init__(self, filename):

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def unpack(self, *message):
        mes_str = ' '.join(map(str, *message))
        return mes_str

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def info(self, *message):
        mes_str = self.unpack(message)
        self.logger.info(mes_str)
        self._flush()
