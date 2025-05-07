from django.shortcuts import render
from django.views.decorators.cache import cache_control
import os
import joblib

# Load models
model1 = joblib.load(os.path.join(os.path.dirname(__file__), "Random_Forest_model1.pkl"))
model2 = joblib.load(os.path.join(os.path.dirname(__file__), "svm_model1.pkl"))

@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    return render(request, 'index.html')

@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def checkSpam(request):
    if request.method == "POST":
        algo = request.POST.get("algo")
        rawData = request.POST.get("rawdata")

        if algo == "Algo-1":
            result = model1.predict([rawData])[0]
        elif algo == "Algo-2":
            result = model2.predict([rawData])[0]
        else:
            result = "invalid"

        return render(request, 'output.html', {"answer": result})
    else:
        return redirect('/')
