import datetime
from urllib.parse import urlencode
import pandas as pd
import requests
from time import strftime
import time
import numpy as np
import matplotlib  
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm

def kdj(high, low, close, n=9):
    """
    计算KDJ指标
    :param high: 最高价序列
    :param low: 最低价序列
    :param close: 收盘价序列
    :param n: 周期，默认为9
    :return: K线、D线、J线序列
    """
    # 计算未成熟随机值RSV
    max_r = pd.Series(high).rolling(n, min_periods=1).max()
    min_r = pd.Series(low).rolling(n, min_periods=1).min()
    rsv = (pd.Series(close) - min_r) / (max_r - min_r) * 100

    # 计算K值、D值、J值
    k = pd.Series(0.0, index=rsv.index)
    d = pd.Series(0.0, index=rsv.index)
    j = pd.Series(0.0, index=rsv.index)
    for i in range(n, len(rsv)):
        k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
        d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        j[i] = 3 * k[i] - 2 * d[i]
    k_new=[i for i in k if i != 0]
    d_new=[i for i in d if i != 0]
    j_new=[i for i in j if i != 0]
    plt.plot(np.arange(len(k_new)),k_new,label='k linear')
    plt.plot(np.arange(len(d_new)),d_new,label='d linear')
    plt.plot(np.arange(len(d_new)),j_new,label='j linear')
    plt.legend()
    plt.title("KDL linear")
    plt.show()
    print('\nKDJ系统建议（若无输出，则说明暂时还没有可用信息）:')
    if d[len(d)-1]>80:
        print("股票价格已经非常高，并处于超买状态。此时可以考虑卖出，以防股价在短期内下跌造成损失")
    if d[len(d)-1]<20:
        print("股票价格已经非常高，并处于超卖状态。此时可以考虑买入，以防期待股价反弹") 
    if j[len(d)-1]<10:
        print("股票价格已经非常高，并处于超卖状态。此时可以考虑买入，以防期待股价反弹") 
    if j[len(d)-1]>100:
        print("股票价格已经非常高，并处于超买状态。此时可以考虑卖出，以防股价在短期内下跌造成损失")   
    for i in range(len(d)-3,len(d)):
        if k[i-1]<d[i-1] and k[i]>d[i] and (k[i]>70 or k[i]<30 )and (j[i]>70 or j[i]<30):
            print("线K向上突破线D，买进信号")
        if k[i-1]>d[i-1] and k[i]>d[i]and (k[i]>70 or k[i]<30 )and (j[i]>70 or j[i]<30):
            print("线K向下突破线D，卖出信号")
            
    # 注意：KD指标不适于发行量小，交易不活跃的股票；KD指标对大盘和热门大盘股有极高准确性
               

def RSI(df,period):
    df['change'] = df['Close'].diff()
    df['gain'] = df['change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = -df['change'].apply(lambda x: x if x < 0 else 0)

    # 分别计算股票收盘价上涨日和下跌日的平均涨幅和平均跌幅
    
    avg_gain = df['gain'][:period].sum() / period
    avg_loss = df['loss'][:period].sum() / period
    x=np.zeros(len(df))
         # 计算RSI指标的数值
    t=len(df) 
    i=period  
    while i<t:  
        avg_gain = ((period-1) * avg_gain + df.iloc[i]['gain']) / period
        avg_loss = ((period-1) * avg_loss + df.iloc[i]['loss']) / period
        x[i] = avg_gain / avg_loss if avg_loss != 0 else 100
        x[i]=100 - 100 / (1 + x[i]) 
        i+=1     
    #将RS值转化为RSI指标（范围：0~100）
        # 代码中的 avg_gain 和 avg_loss 分别表示股票历史数据中过去一段时间内的平均涨幅和平均跌幅。如果平均跌幅为 0，则将 RS 的值设为 100，以避免出现除数为 0 的情况。最终的计算结果会被存储在 DataFrame 中，其中 i 表示当前的行数
    return x

def gen_secid(rawcode: str) -> str:
    '''
    生成东方财富专用的secid
    Parameters
    ----------
    rawcode : 6 位股票代码
    Return
    ------
    str: 指定格式的字符串
    '''
    # 沪市指数
    if rawcode[:3] == '000':
        return f'1.{rawcode}'
    # 深证指数
    if rawcode[:3] == '399':
        return f'0.{rawcode}'
    # 沪市股票
    if rawcode[0] != '6':
        return f'0.{rawcode}'
    # 深市股票
    return f'1.{rawcode}'

def get_k_history(code: str, beg: str, end: str, klt, fqt: int = 1) -> pd.DataFrame:
    '''
    功能获取k线数据
    参数
        code : 6 位股票代码
        beg: 开始日期 例如 20200101
        end: 结束日期 例如 20200201
        klt: k线间距 默认为 101 即日k
            klt:1 1 分钟
            klt:5 5 分钟
            klt:101 日
            klt:102 周
        fqt: 复权方式
            不复权 : 0
            前复权 : 1
            后复权 : 2 
    '''
    EastmoneyKlines = {
        'f51': 'dates',
        'f52': 'opens',
        'f53': 'closes',
        'f54': 'highs',
        'f55': 'lows',
        'f56': 'volumes',   
    }
    EastmoneyHeaders = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Referer': 'http://quote.eastmoney.com/center/gridlist.html',
    }
    fields = list(EastmoneyKlines.keys())
    columns = list(EastmoneyKlines.values())
    fields2 = ",".join(fields)
    secid = gen_secid(code)
    params = (
        ('fields1', 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13'),
        ('fields2', fields2),
        ('beg', beg),
        ('end', end),
        ('rtntype', '6'),
        ('secid', secid),
        ('klt', f'{klt}'),
        ('fqt', f'{fqt}'),
    )
    params = dict(params)
    base_url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    url = base_url+'?'+urlencode(params)
    json_response: dict = requests.get(
        url, headers=EastmoneyHeaders).json()

    data = json_response.get('data')
    if data is None:
        if secid[0] == '0':
            secid = f'1.{code}'
        else:
            secid = f'0.{code}'
        params['secid'] = secid
        url = base_url+'?'+urlencode(params)
        json_response: dict = requests.get(
            url, headers=EastmoneyHeaders).json()
        data = json_response.get('data')
    if data is None:
        print('股票代码:', code, '可能有误')
        return pd.DataFrame(columns=columns)
    klines = data['klines']
    rows = []
    for _kline in klines:
        kline = _kline.split(',')
        rows.append(kline)
    date = pd.DataFrame(rows, columns=columns)
    return date

if __name__ == "__main__":
    # 股票代码600276
    code = input("请输入你要咨询的股票代码（测试时，可以使用代码：600276）\n")
    filename=code+'.csv'
    ddd=111
    while ddd!=000:  
        ddd=int(input('请输入您需要的的类型\n(1)短期分析（日K图）：101\n(2)中期分析（周K图）：102\n(3)长期分析（月K图）：103\n输入000结束程序\n'))
        if ddd!=101 and ddd!=102 and ddd!=103 and ddd!=000:
            print('输入有误，请重新输入\n')
            continue
        if ddd==000:
            break
        now=datetime.datetime.now()
        weekday=now.weekday()
        # 结束日期 
        end_date = time.strftime("%Y%m%d", time.localtime())
        # 开始日期
        start_date='20230411'
        if ddd==101:
            start_date = str(int(end_date)-200)
        if ddd==102:
            start_date = str(int(end_date)-10000)
        if ddd==103:
            end_date = time.strftime("%Y", time.localtime())
            end_date1 = time.strftime("%m", time.localtime())
            end_date=end_date+end_date1+'01'
            start_date = str(int(end_date)-30000) 
        print(f'正在获取 {code} 从 {start_date} 到 {end_date} 的 k线数据......')
        # 根据股票代码、开始日期、结束日期获取指定股票代码指定日期区间的k线数据
        data = get_k_history(code, start_date, end_date,klt=ddd)
        # 保存k线数据到表格里面
        data.to_csv(f'{code}.csv', encoding='utf-8-sig', index=None)
        print(f'股票代码：{code} 的 k线数据已保存到代码目录下的 {code}.csv 文件中')

        #从“东方财富”网站下载数据到与code同目录的csv文件中
        
        data=pd.read_csv(filename,index_col=0)
        data.rename(columns={'dates':'Date','opens':'Open','highs':'High','lows':'Low','closes':'Close','volumes':'Volume'},inplace=True)
        my_color = mpf.make_marketcolors(up='red',
                                        down='green',
                                        edge='inherit',
                                        wick='inherit',
                                        volume='inherit')
        # 设置图表的背景色
        my_style = mpf.make_mpf_style(marketcolors=my_color,
                                    gridcolor='(0.82, 0.83, 0.85)')
        # 读取的测试数据索引为字符串类型，需要转化为时间日期类型
        data.index = pd.to_datetime(data.index)
        mpf.plot(data, style=my_style, type='candle', volume=True)
        '''
        葛兰碧八大法则的主要思想就是利用短期均线与长期均线的关系，找出其中蕴含的信息
        '''
        if ddd==101:#短期判断
            data['Ma5'] = data.Close.rolling(window=5).mean()  # 求5日均线
            data['Ma10'] = data.Close.rolling(window=10).mean()  # 求10日均线
            data['Ma15'] = data.Close.rolling(window=15).mean()  # 求15日均线
            k11, b11 = np.polyfit([1,2,3,4,5,6], [data.Close.iloc[-6],data.Close.iloc[-5],data.Close.iloc[-4],data.Close.iloc[-3],data.Close.iloc[-2],data.Close.iloc[-1]], 1)#最小二乘拟合，判断股价变化趋势（此种方法不严谨但可以在一定程度上反映数据变化趋势）
            k12,b12=np.polyfit([1,2,3,4,5],[data.Volume.iloc[-6],data.Volume.iloc[-5],data.Volume.iloc[-4],data.Volume.iloc[-3],data.Volume.iloc[-2]],1)
            if  data.Close.iloc[-2] > data.iloc[-2]['Ma5']and \
                data.iloc[-2]['Ma5'] > data.iloc[-2]['Ma10']and \
                data.iloc[-2]['Ma10'] > data.iloc[-2]['Ma15']and \
                data.Close.iloc[-3] > data.iloc[-3]['Ma5']and \
                data.iloc[-3]['Ma5'] > data.iloc[-3]['Ma10']and \
                data.iloc[-3]['Ma10'] > data.iloc[-3]['Ma15']and\
                data.Close.iloc[-4] > data.iloc[-4] ['Ma5']and\
                data.iloc[-4]['Ma5'] > data.iloc[-4]['Ma10']:
                    # 这段条件写的有点冗长了，可以再简化一下。条件（1）判断短期均线突破长期与否，条件（2）判断近期股价的变化趋势，条件（3）判断量价关系（量增价涨为买入信号）
                print('根据葛兰碧八大法则：')    
                print('\n从此前三日开始，股价突破五日均线，\n五日均线突破十日均线，十日均线突破\n十五日均线（均未掉落且有上升趋势）,\n所以预期短期内该股票价格将会呈现上升趋势，\n建议买进该股票\n')
                if k11*k12>0:
                    print('根据量价关系：')
                    print('量增价高，建议买入')    
            else:
                print('均线系统建议短期不建议此时买进该股票(推荐看看中长期预测)')
            kdj(data.High,data.Low,data.Close)
            plt.plot(data.Close,label='1-day moving averange') 
            plt.plot(data['Ma5'],'r',label='5-day moving averange')
            plt.plot(data['Ma10'],'g',label='10-day moving averange')
            plt.plot(data['Ma15'],'b',label='15-day moving averange')
            plt.xticks(rotation=30)
            plt.legend()
            plt.title("Moving averange system")
            plt.show()
        if ddd==102:#中期判断
            data['Ma5'] = data.Close.rolling(window=5).mean()  # 求5周均线
            data['Ma10'] = data.Close.rolling(window=10).mean()  # 求10周均线
            data['Ma15'] = data.Close.rolling(window=15).mean()  # 求15周均线
            k21, b21 = np.polyfit([1,2,3,4,5,6], [data.Close.iloc[-6],data.Close.iloc[-5],data.Close.iloc[-4],data.Close.iloc[-3],data.Close.iloc[-2],data.Close.iloc[-1]], 1)
            k22,b22=np.polyfit([1,2,3,4],[data.Volume.iloc[-5],data.Volume.iloc[-4],data.Volume.iloc[-3],data.Volume.iloc[-2]],1)
            if  data.Close.iloc[-2] > data.iloc[-2]['Ma5']and \
                data.iloc[-2]['Ma5'] > data.iloc[-2]['Ma10']and \
                data.iloc[-2]['Ma10'] > data.iloc[-2]['Ma15']and \
                data.Close.iloc[-3] > data.iloc[-3]['Ma5']and \
                data.iloc[-3]['Ma5'] > data.iloc[-3]['Ma10']and \
                data.iloc[-3]['Ma10'] > data.iloc[-3]['Ma15']and\
                data.Close.iloc[-4] > data.iloc[-4] ['Ma5']and\
                data.iloc[-4]['Ma5'] > data.iloc[-4]['Ma10']:
                print('根据葛兰碧八大法则：')
                print('\n从此前三周开始，股价突破五周均线，\n五周均线突破十周均线，十周均线突破\n十五周均线（均未掉落且有上升趋势），所以预期\n中期内该股票价格将会呈现上升趋势，\n建议买进该股票\n')
                if k21*k22>0:
                    print('根据量价关系：')
                    print('量增价高，建议买入')    
            else:
               print('均线系统建议中期不要买进该股票')
            plt.figure(figsize=(7,5.5))
            plt.plot(data.Close,label='1-week moving averange')
            plt.plot(data['Ma5'],'r',label='5-week moving averange')
            plt.plot(data['Ma10'],'g',label='10-week moving averange')
            plt.plot(data['Ma15'],'b',label='15-week moving averange')
            plt.xticks(rotation=30)
            plt.legend()
            plt.title("Moving averange system")
            plt.show()
            x=RSI(data,14)#将时间段选择为过去14个交易日,快速RSI
            new_x = [i for i in x if i != 0]
            y=RSI(data,21)#将时间段选择为过去21个交易日,慢速RSI
            new_y= [i for i in y if i != 0]
            plt.plot(np.arange(len(new_x)), new_x,label='fast_RSI')
            plt.plot(np.arange(7,len( new_y)+7), new_y,label='slow_RSI')
            plt.title("RSI linear")
            plt.legend()
            plt.show()
            t=len(x)-1
            print("RSI指标反映出：") 
            for i in range(t-2,t):
                if x[i]<20 and y[i]<20 and x[i-1]<y[i-1] and x[i]>y[i]:
                    print('近期，快速RSI在20以下水平由下往上交叉慢速RSI，是买入信号；')
                else:
                    print('往前',end='')
                    print(t-i,end='')
                    print('周，暂未发现快速RSI在20以下水平由下往上交叉慢速RSI，无买入信号')
            if x[t]>=50 and x[t]<80:
                    print("50 < RSI < 80，说明股票价格处于相对强势状态，可以考虑买入并持仓以期待股价继续上涨。")
            elif x[t]>=80 and x[t]<100:    
                print("80 < RSI < 100说明股票价格已经非常高，并处于超买状态。此时可以考虑卖出，以防股价在短期内下跌造成损失。")
            elif x[t]>=0 and x[t]<20:
                print("0 < RSI < 20，说明股票价格已经非常低，并处于超卖状态。此时可以视为买入时机，以期在股价反弹时获得收益")
            elif  x[t]>=20 and x[t]<50:   
                print("20 < RSI < 50 说明股票价格呈现弱势走势，可能会继续下跌。此时建议空仓观望，等待更好的买入时机\n")
            #当0 < RSI < 20 时， 极弱，超卖，买入；
            #当20 < RSI < 50 时， 弱势，卖出，空仓；
            #当50 < RSI < 80 时， 强势，买入，持仓；
            #当80 < RSI < 100时， 极强，超买，卖出。
            kdj(data.High,data.Low,data.Close)
        if ddd==103:#长期
            data['Ma5'] = data.Close.rolling(window=5).mean()  # 求5月均线
            data['Ma10'] = data.Close.rolling(window=10).mean()  # 求10月均线
            data['Ma15'] = data.Close.rolling(window=15).mean()  # 求15月均线
            k31, b31 = np.polyfit([1,2,3,4,5,6], [data.Close.iloc[-6],data.Close.iloc[-5],data.Close.iloc[-4],data.Close.iloc[-3],data.Close.iloc[-2],data.Close.iloc[-1]], 1)
            #利用最小二乘法拟合进几个数据点的斜率来判断趋势
            k32,b32=np.polyfit([1,2,3,4],[data.Volume.iloc[-4],data.Volume.iloc[-3],data.Volume.iloc[-2],data.Volume.iloc[-1]],1)
            print(k31,k32)
            if  data.Close.iloc[-2] > data.iloc[-2]['Ma5']and \
                data.iloc[-2]['Ma5'] > data.iloc[-2]['Ma10']and \
                data.iloc[-2]['Ma10'] > data.iloc[-2]['Ma15']and \
                data.Close.iloc[-3] > data.iloc[-3]['Ma5']and \
                data.iloc[-3]['Ma5'] > data.iloc[-3]['Ma10']and \
                data.iloc[-3]['Ma10'] > data.iloc[-3]['Ma15']and\
                data.Close.iloc[-4] > data.iloc[-4] ['Ma5']and\
                data.iloc[-4]['Ma5'] > data.iloc[-4]['Ma10']:
                print('根据葛兰碧八大法则：')       
                print('\n从此前三月开始，股价突破五月均线，\n五月均线突破十月均线，十月均线突破\n十五月均线（均未掉落且有上升趋势），所以预期\n长期内该股票价格将会呈现上升趋势，\n建议买进该股票\n')
                if k31*k32>0:
                    print('根据量价关系：')
                    print('量增价高，建议买入')
            else:
                print('均线系统建议长期不要买进该股票')
            plt.figure(figsize=(7,5.5))
            plt.plot(data.Close,label='1-month moving averange')
            plt.plot(data['Ma5'],'r',label='5-month moving averange')
            plt.plot(data['Ma10'],'g',label='10-month moving averange')
            plt.plot(data['Ma15'],'b',label='15-month moving averange')
            plt.xticks(rotation=30)
            plt.legend()
            plt.title("Moving averange system")
            plt.show()
            x=RSI(data,14)#将时间段选择为过去14个交易日,快速RSI
            new_x = [i for i in x if i != 0]
            y=RSI(data,21)#将时间段选择为过去21个交易日,慢速RSI
            new_y= [i for i in y if i != 0]
            plt.plot(np.arange(len(new_x)), new_x,label='fast_RSI')
            plt.plot(np.arange(7,len( new_y)+7), new_y,label='slow_RSI')
            plt.title("RSI linear")
            plt.legend()
            plt.show()
            t=len(x)-1
            print("RSI指标反映出：") 
            for i in range(t-2,t):
                if x[i]<20 and y[i]<20 and x[i-1]<y[i-1] and x[i]>y[i]:
                    print('近期，快速RSI在20以下水平由下往上交叉慢速RSI，是买入信号；')
                else:
                    print('往前',end='')
                    print(t-i,end='')
                    print('月，暂未发现快速RSI在20以下水平由下往上交叉慢速RSI，无买入信号')
        
            if x[t]>=50 and x[t]<80:
                    print("50 < RSI < 80，说明股票价格处于相对强势状态，可以考虑买入并持仓以期待股价继续上涨。")
            elif x[t]>=80 and x[t]<100:    
                print("80 < RSI < 100说明股票价格已经非常高，并处于超买状态。此时可以考虑卖出，以防股价在短期内下跌造成损失。")
            elif x[t]>=0 and x[t]<20:
                print("0 < RSI < 20，说明股票价格已经非常低，并处于超卖状态。此时可以视为买入时机，以期在股价反弹时获得收益")
            elif  x[t]>=20 and x[t]<50:   
                print("20 < RSI < 50 说明股票价格呈现弱势走势，可能会继续下跌。此时建议空仓观望，等待更好的买入时机\n")
            #当0 < RSI < 20 时， 极弱，超卖，买入；
            #当20 < RSI < 50 时， 弱势，卖出，空仓；
            #当50 < RSI < 80 时， 强势，买入，持仓；
            #当80 < RSI < 100时， 极强，超买，卖出。

print('\n感谢您对本程序的使用')
print('注意：本程序只基于葛兰碧法则，RSI指标，KDJ指标为用户提供建议，对客户自身的损失不承担任何责任')











