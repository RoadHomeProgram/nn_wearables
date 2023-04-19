import matplotlib.pyplot as plt
from datetime import datetime

def ppgVsSurvey(ppg,mood):
    plt.plot(ppg['ts'],ppg['ir'])
    plt.axvline(x=datetime.strptime(mood['date'],'%m/%d/%Y %H:%M:%S %Z'))
    return plt