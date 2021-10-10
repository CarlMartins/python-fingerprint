from django.shortcuts import render


def index(request):
    """Placeholder index view"""
    return render(request, )


def teste(request):
    return render(request,'teste.html')