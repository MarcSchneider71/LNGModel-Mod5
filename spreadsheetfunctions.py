import pandas as pd

import qgrid

def makeadataframewithmutiplesheet(xml):
    full_table = pd.DataFrame()
    for name, sheet in xml.items():
        sheet['sheet'] = name
        sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
        full_table = full_table.append(sheet)
        full_table.reset_index(inplace=True, drop=True)
    return full_table
