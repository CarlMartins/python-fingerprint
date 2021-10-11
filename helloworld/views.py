import os.path

from django.shortcuts import render,redirect
from django.http import HttpResponse


def index(request):
    return render(request,'index.html')

def login(request):
    query = request.GET.get('teste')
    return render(request,'login.html')

def start(request):
    return redirect('/login')