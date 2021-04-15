from tifffile import imread
from matplotlib import pyplot as plt
from .util import printProgressBar as progress
import numpy as np
import os
import ntpath


class Image(object):
    def __init__(self, path):
        self.path = path
        self.filename = ntpath.basename(path)
        self.band = self.filename.split('_')[0]
        self.loaded = None

    def get(self, key):
        return self.__getattribute__(key)

    def load(self):
        if self.loaded is None:
            self.loaded = imread(str(self.path))

        return self.loaded

    def show(self):
        plt.imshow(self.load())
        plt.show()


class FieldImages(object):
    def __init__(self, path):
        self.path = path

        bands = []
        for image in os.listdir(path):
            if os.path.isfile(os.path.join(path, image)):
                bands.append(image.split('_')[1])
                setattr(self, bands[-1], Image(os.path.join(self.path, image)))

        bands.sort()
        self.bands = np.array(bands)

    def get(self, key):
        return self.__getattribute__(key)

    def show_rgb(self):
        rgb = np.dstack((self.B04.load(), self.B03.load(), self.B02.load()))
        plt.imshow(rgb * 3)
        plt.show()


class Dates(object):
    def __init__(self, path):
        self.path = path
        self.X = AttributeError('Attribute not loaded. To load all data call (ModelData).load_all_data()')
        self.y = AttributeError('Attribute not loaded. To load all data call (ModelData).load_all_data()')

        dates = []
        for date in os.listdir(path):
            if os.path.isdir(os.path.join(path, date)):
                date = f'D{date}'
                dates.append(date)
                setattr(self, date, FieldImages(os.path.join(self.path, date[1:])))
            elif os.path.isfile(os.path.join(path, date)) and 'label' in date:
                self.label = imread(os.path.join(path, date))
            elif os.path.isfile(os.path.join(path, date)) and 'id' in date:
                self.field_id = imread(os.path.join(path, date))

        dates.sort()
        self.dates = np.array(dates)

    def get(self, key):
        return self.__getattribute__(key)

    def load_group_data(self):
        self.y = {'label': self.label, 'field_id': self.field_id}
        self.X = {}

        for date in self.dates:
            d = self.get(date)

            for band in d.bands:
                if band not in self.X:
                    self.X[band] = []
                progress(len(self.X[band]), len(self.dates), prefix=f'Downloaded {len(self.X[band])} / {len(self.dates)} dates',)
                self.X[band].append(d.get(band).load())

        progress(1, 1, prefix=f'Downloaded the group.  ',)


class ModelData(object):
    def __init__(self, path):
        self.path = path
        self.X = AttributeError('Attribute not loaded. To load all data call (ModelData).load_all_data()')
        self.y = AttributeError('Attribute not loaded. To load all data call (ModelData).load_all_data()')

        groups = []
        for group in os.listdir(path):
            if os.path.isdir(os.path.join(path, group)):
                group = f'G{group}'
                groups.append(group)
                setattr(self, group, Dates(os.path.join(self.path, group[1:])))

        groups.sort()
        self.groups = np.array(groups)

    def get(self, key):
        return self.__getattribute__(key)

    def load_all_data(self):
        self.y = []
        self.X = []

        for group in self.groups:
            g = self.get(group)
            self.y.append({})
            self.y[-1]['label'] = g.label
            self.y[-1]['field_id'] = g.field_id
            images = dict()

            for date in g.dates:
                d = g.get(date)

                for band in d.bands:
                    if band not in images:
                        images[band] = []
                    progress(len(images[band]), len(g.dates),
                             prefix=f'Downloading {len(self.y)} / {len(self.groups)} groups',
                             suffix=f'Downloading {len(images[band])} / {len(g.dates)} dates in group {group}  ')
                    images[band].append(d.get(band).load())

            self.X.append(images)

        progress(1, 1, prefix='Downloaded all groups.', suffix='Downloaded all images.                 ')

