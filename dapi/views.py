
# Create your views here.
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# from django.shortcuts import render
# from rest_framework import generics
# from .serializers import PostSerializer

@method_decorator(csrf_exempt, name="dispatch")
def certification(request):
    import cv2
    import base64
    from PIL import Image
    from io import BytesIO
    import numpy as np

    #학습된 모델 호출(혁중)
    model = cv2.face.LBPHFaceRecognizer_create() 
    model.read('trainer.yml')


    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

    def face_detector(img, size = 0.5): 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = face_classifier.detectMultiScale(gray,1.3,5) 
        if faces is(): 
            return img,[] 
        for(x,y,w,h) in faces: 
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2) 
            roi = img[y:y+h, x:x+w] 
            roi = cv2.resize(roi, (200,200)) 
        return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달 


    # 얼굴 사진 수정
    # 사진 받을 post key값 "img" : src
    # cv2.COLOR_BGR2RGB
    def stringToRGB(base64_string):
        imgdata = base64.b64decode(base64_string)
        dataBytesIO = BytesIO(imgdata)
        image = Image.open(dataBytesIO)
        return cv2.cvtColor(np.array(image), cv2.IMREAD_COLOR)

    src = request.POST["img"].split("base64,")[1]

    image, face = face_detector(stringToRGB(src)) 

    try: 
        #검출된 사진을 흑백으로 변환  
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) 
        result = model.predict(face) 
        if result[1] < 500:  
            confidence = int(100*(1-(result[1])/300)) 
        if confidence > 85:
            return JsonResponse({"result":True}, safe=False)
        else: 
            return JsonResponse({"result":False}, safe=False)
        
    except: 
        #얼굴 검출 안됨  
        return JsonResponse({"result":"얼굴 검출 불가"}, safe=False)

@method_decorator(csrf_exempt, name="dispatch")
def pred(request):
    import numpy as np
    import pickle
    import pandas as pd

    # print("=============================")
    # print(request.POST.get("age"))
    # print("=============================")

    filePath2 = 'model.pkl'
    post_list = ['age', 'income', 'annual', 'job', 'issue', 'family']
    values = []

    model = pickle.load(open(filePath2, 'rb'))

    for post in post_list:
        values.append(float(request.POST.get(post)))


    power = (values[1]/((values[0]*365)+(values[2]*365)))/130

    f_income = np.log((values[1]/ values[-1])/130) 
 

    values.insert(0, power)
    values.insert(2,f_income)
    values[3]= np.log(values[3])

    input_features = pd.Series(values,index=['능력', '나이', '가족평균수입', '연간소득', '연차', '직업유형', '카드발급년수', '가족수']) 
    input_features_df = pd.DataFrame(input_features)
    input_features_df_T = input_features_df.T
    grade =  model.predict(input_features_df_T)
    grade = int(grade)
    context={
        'result': grade
    }

    return JsonResponse(context, safe=False)
    # return render(request, 'result.html', context)

# def send(request):
#     return render(request, 'send.html')