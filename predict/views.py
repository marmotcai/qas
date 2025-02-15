from django.shortcuts import render

# coding:utf-8

import json
import os
import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from datetime import datetime as dt
from predict import models
from .models import Company

import trendanalysis as ta
from trendanalysis.core import manager as my_man

LOCAL = False

def get_hist_predict_data(stock_code):
    recent_data, predict_data = None, None
    # company = models.Company.objects.get(stock_code=stock_code)
    company = get_object_or_404(Company, stock_code = stock_code)

    if company.historydata_set.count() <= 0:
        history_data = models.HistoryData()
        history_data.company = company
        history_data.set_data(run.get_hist_data(stock_code = stock_code, recent_day = 20))
        history_data.save()
        recent_data = history_data.get_data()
    else:
        all_data = company.historydata_set.all()
        for single in all_data:
            now = dt.now()
            end_date = single.get_data()[-1][0]
            end_date = dt.strptime(end_date, "%Y-%m-%d")
            if LOCAL & (now.date() > end_date.date()):        # 更新预测数据
                single.set_data(run.get_hist_data(stock_code = stock_code, recent_day = 20))
                single.save()

            recent_data = single.get_data()
            break

    if company.predictdata_set.count() <= 0:
        predict_data = models.PredictData()
        predict_data.company = company
        predict_data.set_data(my_man.prediction(stock_code,pre_len=10))
        predict_data.save()
        predict_data = predict_data.get_data()
    else:
        all_data = company.predictdata_set.all()
        for single in all_data:
            now = dt.now()
            start_date = dt.strptime(single.start_date,"%Y-%m-%d")
            if LOCAL & (now.date() > start_date.date()):  # 更新预测数据
                single.set_data(my_man.prediction(stock_code, pre_len=10))
                single.save()

            predict_data = single.get_data()
            break

    return recent_data, predict_data

def get_crawl_save_data():
    """
    将10个公司的指标数据爬取并保存到数据库
    """
    # 此处应是从网上爬取数据，并保存为csv文件
    parent_dir = os.path.dirname(__file__)  # "predict/views.py"
    file_dir = os.path.join(parent_dir, "stock_index/")
    for file_name in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file_name)
        data_frame = pd.read_csv(file_path)
        stock_code = file_name.split('.')[0]
        company = get_object_or_404(Company, stock_code=stock_code)
        for index,row in data_frame.iterrows():
            company.stockindex_set.create(ri_qi=row['ri_qi'],zi_jin=row['zi_jin'],qiang_du=row['qiang_du'],feng_xian=row['feng_xian'],
                zhuan_qiang=row['zhuan_qiang'],chang_yu=row['chang_yu'],jin_zi=row['jin_zi'],zong_he=row['zong_he'])

def get_stock_index(stock_code):
    """
    获取股票的各项指标数据
    """
    company = get_object_or_404(Company, stock_code=stock_code)
    if company.stockindex_set.count() <= 0:
        # 将爬取的数据存入数据库
        get_crawl_save_data()
    # 从数据库获取近三天的数据
    indexs = company.stockindex_set.all().order_by('-ri_qi')[:3].values()
    return list(indexs)

def home(request):
    recent_data, predict_data = get_hist_predict_data("300096")
    data = {"recent_data":recent_data,"stock_code":"300096","predict_data":predict_data}
    # data['indexs'] = get_stock_index("300096")
    return render(request,"predict/home.html",{"data":json.dumps(data)}) # json.dumps(list)

def predict_stock_action(request):
    stock_code = request.POST.get('stock_code', None)
    ta.g.log.info("request stock_code: " + stock_code)

    recent_data, predict_data = get_hist_predict_data(stock_code)
    data = {"recent_data": recent_data, "stock_code": stock_code, "predict_data": predict_data}
    # data['indexs'] = get_stock_index(stock_code)
    return render(request, "predict/home.html", {"data": json.dumps(data)})  # json.dumps(list)

def index(request):
    stock_code = request.POST.get('stock_code', None)
    if (stock_code == None):
        return render(request, "predict/home.html")

    ta.g.log.info("request stock_code: " + stock_code)

    company = get_object_or_404(Company, stock_code=stock_code)
    history_data = models.HistoryData()
    history_data.company = company
    history_data.set_data(my_man.get_hist_data(code=stock_code, recent_day=20))
    history_data.save()

    recent_data = history_data.get_data()
    predict_data = models.PredictData()
    predict_data.company = company
    predict_data.set_data(my_man.prediction(stock_code))
    predict_data.save()
    predict_data = predict_data.get_data()
    data = {"recent_data": recent_data, "stock_code": stock_code, "predict_data": predict_data}
    # recent_data, predict_data = get_hist_predict_data(stock_code)
    # data = {"recent_data": recent_data, "stock_code": stock_code, "predict_data": predict_data}
    # data['indexs'] = get_stock_index(stock_code)
    return render(request, "predict/home.html", {"data": json.dumps(data)})  # json.dumps(list)

def init_db():
    initcode_file = os.path.join(ta.g.data_path, ta.g.config["data"]["init_codefile"])
    data_frame = pd.read_csv(initcode_file, index_col=False, encoding='UTF-8')
    for index, row in data_frame.iterrows():
        code = "%06d" % int(row['code'])
        Company.objects.create(name=row['name'], stock_code=code)
        print(row['code'], ':', row['name'])

def init(request):
    c = Company.objects.all()
    if (c.count() <= 0):
        init_db()

    return HttpResponse(u"初始化")
