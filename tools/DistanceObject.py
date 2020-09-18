
class DistanceObject(object):

    def __init__(self, additive_dist=0, average_dist_1=0, average_dist_2=0, path=None):

        self._additive_dist = additive_dist
        self._average_dist_1 = average_dist_1
        self._average_dist_2 = average_dist_2
        self._path = path

    @property
    def additive_dist(self):
        return self._additive_dist

    @additive_dist.setter
    def additive_dist(self, value):
        self._additive_dist = value

    @property
    def average_dist_1(self):
        return self._average_dist_1

    @average_dist_1.setter
    def average_dist_1(self, value):
        self._average_dist_1 = value

    @property
    def average_dist_2(self):
        return self._average_dist_2

    @average_dist_2.setter
    def average_dist_2(self, value):
        self._average_dist_2 = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value
