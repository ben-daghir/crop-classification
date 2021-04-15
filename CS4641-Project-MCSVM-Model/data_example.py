# File Structure Break Down
# ├── GROUP00
# │   ├── field_id.tif
# │   ├── label.tif
# │   │
# │   ├── DATE20190606
# │   │   ├── BAND01.tif
# │   │   ├── ...
# │   │   └── CLD.tif
# │   └── DATE...
# │       ├── ...
# │       └── BAND...
# │
# ├── GROUPXX
# │   ├── FIELD_ID
# │   ├── LABEL
# │   │
# │   └── DATEXXXXXXXX
# │   :   :
# │   :   └─ BANDXXX
# :   :
#
# # ModelData Indexing Format
# /GROUP00/DATE20190606/BAND01.tif --> data.G00.D20190606.B01 (Object)
# /GROUP00/DATE20190606/CLD.tif --> data.G00.D20190606.CLD (Object)

# Example
from utils.data import ModelData
from pathlib import Path
path = str(Path.home()) + '/Dropbox (GaTech)'

data = ModelData(path)

# To create an object for the image /YOUR/PATH/TO/Dropbox (GaTech)/CS4641/data/00/DATE20190606/0_B01_20190606.tif
image_object = data.G00.D20190606.B01

# To physically load the image array
image_array = image_object.load()

# To plot/show the image
image_object.show()

# To show an rgb version of a group for a specific date. (A combination of bands [B04, B03, B02] => [R, G, B])
date_object = data.G00.D20190606
date_object.show_rgb()

# Iterating over all Groups, Dates, and Bands
groups = data.groups
dates = data.get(groups[0]).dates
bands = data.get(groups[0]).get(dates[0]).bands

print(f'Groups: {groups}')
print(f'Dates in group {groups[0]}: {dates}')
print(f'Bands in group {groups[0]} and date {dates[0]}: {bands}')

# ModelData does not load any of the image pixel values until told to do so. The pixel data can be loaded for either
# a single group, or for the entire data set.
#
# The loaded data takes on the following form:
#
# X = [G00,                      ....G01,                  G02,                  G03]
#      / \
#   {'B01': [All band 1 data
#            in chronological
#            order],
#
#    'B02': [All band 2 data
#            in chronological   ==> [np.array([width by height image array]),
#            order],                                ......
#     ...                            np.array([width by height image array])]
#    'CLD': [All cloud data
#            in chronological
#            order]
#    }
#
# y = [G00,                       ....G01,                  G02,                  G03]
#      / \
#     {'label': np.array([G00 Label Image]),
#      'field_id': np.array([G00 Field ID Image])
#     }
#
# If you were to only load the data for one group, then X and y would contain only single group entry from the above
# X and y arrays. For example, just loading the Group 02 data would be done with data.G02.load_group_date().

# # Examples of Preparing Data for your model
data = ModelData(path)

# Load just a single group
g00 = data.G00
g00.load_group_data()

# Accessing the data
Xdata = g00.X
ydata = g00.y
print('\n***Group 00 Data***', '\nX keys:', Xdata.keys(), '\ny keys:', ydata.keys())
print('Example image array for the first date of the B01 band:\n', Xdata['B01'][0])
print('Example label array:\n', ydata['label'])

# Load the entire data set for all groups
data.load_all_data()

# Accessing the data
Xdata = data.X
ydata = data.y
print('\n***All Data***', '\nLength of X:', len(Xdata), '\nLength of y:', len(ydata))
print('X keys at first index (G00):', Xdata[0].keys(), '\ny keys at first index:', ydata[0].keys())
print('Example image array in G00 for the first date of the B01 band:\n', Xdata[0]['B01'][0])
print('Example label array for G00:\n', ydata[0]['label'])

