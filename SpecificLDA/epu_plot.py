import pandas as pd


# 绘制EPU随时间变化的图像
def epu_plot(params, roll, figsize = (10,4)):
    
    def get_data(param):
        data = pd.DataFrame(param['date'],columns = ['date'])
        data['z'] = param['z']
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index(['date'])
        data = data.sort_index()
        day_list = pd.date_range(data.index[0],data.index[-1],freq='D').strftime("%Y-%m-%d").tolist()
        return data, day_list
    
    def get_ts(data):
        ts = []
        for d in data['z']:
            if d == []:
                ts.append(1)
            else:
                ts.append(sum(d)/len(d))
        data['ts'] = ts
        ts_r = []
        for d in data['z']:
            if d == []:
                ts_r.append(0)
            else:
                ts_r.append(1-sum(d)/len(d))
        data['ts_r'] = ts_r
        return data
    
    def getDataDay(data, day_list):
        data_day = pd.DataFrame(columns=['part'])
        for day in day_list:
            day_text = []
            try:
                if type(data.loc[day,'z']) == list:
                    day_text = data.loc[day,'z']
                else:
                    for text in data.loc[day,'z']:
                        day_text.extend(text)
                data_day.loc[day] = 1-sum(day_text)/len(day_text)
            except KeyError:
                data_day.loc[day] = 0
        data_day.columns = ['EPU']
        return data_day
    
    # 绘制图像
    data, day_list = get_data(params)
    data = get_ts(data)
    epu_day = getDataDay(data, day_list)
    epu_day.rolling(roll).mean().plot(figsize=figsize)
    return epu_day

