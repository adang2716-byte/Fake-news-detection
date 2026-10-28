from django.shortcuts import render
from .ml_model import predict_news


def home(request):
    return render(request, "newsapp/index.html")


def result(request):
    if request.method == "POST":
        user_input = request.POST.get("query", "")

        prediction = predict_news(user_input)

        print("INPUT:", user_input)
        print("PREDICTION:", prediction)
        print("TYPE:", type(prediction))

        if prediction.lower() == "fake":
            result_text = "Fake News ❌"
        else:
            result_text = "Real News ✅"

        return render(
            request,
            "newsapp/result.html",
            {
                "result": result_text,
                "text": user_input
            }
        )

    return render(request, "newsapp/result.html")