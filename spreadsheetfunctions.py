import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import os

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import qgrid

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import folium
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)

from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pylab import rcParams
import re
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)
from pandas import Grouper
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler



from datetime import timedelta

from fbprophet import Prophet as proph

from fbprophet.diagnostics import performance_metrics

from fbprophet.diagnostics import cross_validation

from fbprophet.plot import plot_cross_validation_metric

from sklearn import preprocessing

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import folium
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)

from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pylab import rcParams
import re
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)
from pandas import Grouper
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# %matplotlib inline


def makeadataframewithmutiplesheet(xml):
    full_table = pd.DataFrame()
    for name, sheet in xml.items():
        sheet['sheet'] = name
        sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
        full_table = full_table.append(sheet)
        full_table.reset_index(inplace=True, drop=True)
    return full_table
