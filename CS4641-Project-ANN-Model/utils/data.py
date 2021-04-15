from tifffile import imread
from matplotlib import pyplot as plt
from .util import printProgressBar as progress
import numpy as np
import os
import pandas as pd
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
                date = 'D{date}'.format(date)
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

        for dind, date in enumerate(self.dates):
            d = self.get(date)
            for b, band in enumerate(d.bands):
                image = d.get(band).load()
                if band not in self.X:
                    self.X[band] = np.empty((len(self.dates), image.shape[0], image.shape[1]))
                progress(dind, len(self.dates), prefix='Downloaded {dind} / {len(self.dates)} \
                        dates'.format(dind, len(self.dates)))
                self.X[band][dind] = image

        progress(1, 1, prefix='Downloaded the group.')

    def extract_group_fids(self):
        if type(self.y) == AttributeError or type(self.X) == AttributeError:
            self.load_group_data()

        row_locs = []
        col_locs = []
        field_ids = []
        labels = []
        fid_arr = self.y['field_id']
        lab_arr = self.y['label']
        for row in range(fid_arr.shape[0]):
            progress(row, fid_arr.shape[0], prefix='Extracting Field Ids', suffix='Row {row} / \
                    {fid_arr.shape[0]}'.format(row, fid_arr.shape[0]))
            for col in range(fid_arr.shape[1]):
                if fid_arr[row][col] != 0:
                    row_locs.append(row)
                    col_locs.append(col)
                    field_ids.append(fid_arr[row][col])
                    labels.append(lab_arr[row][col])

        X = []
        for i, j in zip(row_locs, col_locs):
            band_data = []
            for band in self.X.keys():
                band_data.append(self.X[band][:, i, j])
            X.append(band_data)

        y = np.zeros((len(X), 8))
        y[range(len(X)), labels] = 1

        progress(1, 1, prefix='Finished Data Extraction.', suffix=' '*15)

        return pd.DataFrame({
                   'fid': field_ids,
                   'label': labels,
                   'i': row_locs,
                   'j': col_locs,
                   'X': X,
                   'y': labels,
                   'y_onehot': list(y)
               })


class ModelData(object):
    def __init__(self, path):
        self.path = path
        self.X = AttributeError('Attribute not loaded. To load all data call (ModelData).load_all_data()')
        self.y = AttributeError('Attribute not loaded. To load all data call (ModelData).load_all_data()')

        groups = []
        for group in os.listdir(path):
            if os.path.isdir(os.path.join(path, group)):
                group = 'G{group}'.format(group)
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
            g.load_group_data()
            self.y.append(g.y)
            self.X.append(g.X)
            # self.y[-1]['label'] = g.label
            # self.y[-1]['field_id'] = g.field_id
            # images = dict()
            # for dind, date in enumerate(g.dates):
            #     d = g.get(date)
            #     for band in d.bands:
            #         image = d.get(band).load()
            #         if band not in images:
            #             images[band] = np.empty((len(g.dates), image.shape[0], image.shape[1]))
            #         progress(dind, len(g.dates),
            #                  prefix=f'Downloading {len(self.y)} / {len(self.groups)} groups',
            #                  suffix=f'Downloading {dind} / {len(g.dates)} dates in group {group}  ')
            #         images[band][dind] = image

        # progress(1, 1, prefix='Downloaded all groups.', suffix='Downloaded all images.                 ')

    def extract_fids(self):
        df = pd.DataFrame()
        for group in self.groups:
            df = df.append(self.get(group).extract_group_fids())
        return df

        # if type(self.y) == AttributeError:
        #     self.load_all_data()
        #
        # row_locs = []
        # col_locs = []
        # field_ids = []
        # labels = []
        # groups = []
        # for group in range(len(self.groups)):
        #     fid_arr = self.y[group]['field_id']
        #     lab_arr = self.y[group]['label']
        #     for row in range(fid_arr.shape[0]):
        #         progress(row, len(self.groups),
        #                  prefix=f'Extracting Group {group} / {len(self.groups)} groups',
        #                  suffix=f'Reading Row {row} / {fid_arr.shape[0]} dates in group {group}  ')
        #         for col in range(fid_arr.shape[1]):
        #             if fid_arr[row][col] != 0:
        #                 row_locs.append(row)
        #                 col_locs.append(col)
        #                 field_ids.append(fid_arr[row][col])
        #                 labels.append(lab_arr[row][col])
        #                 groups.append(group)
        #
        # X = []
        # for i, j, g in zip(row_locs, col_locs, self.groups):
        #     band_data = []
        #     for band in self.X[int(g[-1])].keys():
        #         band_data.append(self.X[int(g[-1])][band][:, i, j])
        #     X.append(band_data)
        #
        # y = np.zeros((len(X), 8))
        # y[range(len(X)), labels] = 1
        #
        # progress(1, 1, prefix=f'Extraction Done.', suffix=' '*10)
        #
        # return pd.DataFrame({
        #            'fid': field_ids,
        #            'label': labels,
        #            'i': row_locs,
        #            'j': col_locs,
        #            'X': X,
        #            'y': labels,
        #            'y_onehot': list(y)
        #        })


