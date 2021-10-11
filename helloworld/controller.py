from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def logar(request):
    name = request.POST.get('user')
    passw = request.POST.get('pass')
    return None