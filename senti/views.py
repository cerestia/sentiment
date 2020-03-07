import jieba
from django.http import HttpResponseRedirect
from django.shortcuts import render
#from senti.tasks import add
from keras.models import load_model
from .models import NLP
def index(request):
    return render(request, 'input.html')

def analyse(request):
    sentence = request.POST['sentence']
    kmodel = NLP()
    res = kmodel.preSen(sentence)
    label_dic = {0: "消极的", 1: "中性的", 2: "积极的"}
    senti = {"senti":label_dic.get(res)}
    return render(request, "result.html", senti)
