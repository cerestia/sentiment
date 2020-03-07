import jieba
from django.http import HttpResponseRedirect
from django.shortcuts import render
#from senti.tasks import add
from keras.models import load_model
from .models import NLP
def index(request):
    return render(request, 'input.html')

def analyse(request):
    kmodel = NLP()
    res = kmodel.preSen("这个东西还不错")

    senti = {"senti":res}
    return render(request, "result.html", senti)
