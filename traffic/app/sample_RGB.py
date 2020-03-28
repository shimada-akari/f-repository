from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
from tensorflow.keras.models import model_from_json
from werkzeug import secure_filename
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)




def img_pred(image):

    #モデルの読み込み
    model = model_from_json(open('AI_standard_32_all_RGB.json', 'r').read())

    # 重みの読み込み
    model.load_weights('AI_standard_32_all_RGB_weight.hdf5')

    image_size = 32

    #image = img_to_array(image)
    image = image.convert("RGB")
    #image = image.resize((image_size, image_size))
    

    

    data = np.asarray(image)
    X = np.array(data)
    X = X.astype('float32')
    X[:, :, 0] -= 100
    X[:, :, 1] -= 116.779
    X[:, :, 2] -= 123.68
    X = cv2.GaussianBlur(X, (5, 5), 0)
    X = X / X.max()
    X = X[None, ...]

    result = np.argmax(model.predict(X))






    
    return result



    


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files['image']:
            img_file = request.files['image']
                
            filename = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = './uploads/' + filename
            return render_template('flask_api_index1.html', img_url=img_url)
        return render_template('./flask_api_index1.html')
    else:
        return render_template('./flask_api_index1.html')


@app.route('/result', methods=['POST'])

def result():
    # submitした画像が存在したら処理する
    if request.files['image']:
        

        # 画像の読み込み
        image_load = load_img(request.files['image'], target_size=(32,32))

        # クラスの予測をする関数の実行
        predict_Confidence = img_pred(image_load)



        japanese = [
            "動物飛び出し注意",
            "警笛鳴らせ",
            "環状交差点",
            "横断禁止",
            "車両侵入禁止",
            "追い越し禁止",
            "一方通行",
            "軌道敷内通行可",
            "ロータリーあり",
            "安全地帯",
            "徐行",
            "止まれ",
            "凹凸注意"
        ]

        meaning = [
            "道路に動物が飛び出すおそれがある(シカ以外にもサルやキツネ、クマなどのものがある)",
            "車両や路面電車は警音器（クラクション）を鳴らさなければいけない",
            "この先に環状交差点がある",
            "この標識より先、歩行者は横断できない",
            "車両(自動車・軽車両・原動機付自転車)の進入ができない",
            "追越しのための右側部分はみ出し通行禁止",
            "車は矢印が示す方向にしか通行できない",
            "自動車が軌道敷内を通行できることを表す標識(補助標識で指定がある場合はその指定の車両のみ)",
            "前方にロータリーがある",
            "標識の位置に、路面電車に乗降する人や、道路を横断する歩行者のために作られた島状の安全地帯がある(車両は進入不可)",
            "この標識より先、車両と路面電車は徐行しなければいけない(徐行とは、すぐに停止が可能な速度)",
            "主に道路交通において、車両などが一時的に停止すること",
            "路面の凹凸があるため車両の運転上注意が必要である",
            
        ]

        eng_meaning = [
            "There is a risk of animals jumping out on the road.",
            "Vehicles and trams must sound horns.",
            "There is a roundabout ahead.",
            "Pedestrians cannot cross.",
            "Vehicles (cars, light vehicles, motorbikes) cannot enter.",
            "Prohibition of overhang on the right side for overtaking.",
            "Vehicles (cars, light vehicles, motorbikes) cannot enter",
            "A sign indicating that the car can pass through the track. (if specified as an auxiliary sign, only the designated car)",
            "There is a roundabout ahead.",
            "here is an island-shaped safety zone made for people getting on and off the tram and pedestrians crossing the road at the sign. (vehicles are not allowed to enter)",
            "Before this sign, the vehicle and the tramway must slow down. (slowing is the speed at which you can stop immediately)",
            "Vehicles temporarily stop mainly in road traffic.",
            "Due to the unevenness of the road surface, care must be taken when driving the vehicle."
        ]

        chi_meaning = [
            "可能会有动物在马路上飞来飞去（除了鹿，还有猴子，狐狸，熊等）",
            "在有此标志的地方，车辆和电车必须发出喇叭声",
            "前方有一个回旋处",
            "行人不能在这个标志之前越过",
            "车辆（汽车，轻型车辆，摩托车）无法进入",
            "禁止在右侧超车",
            "汽车只能沿箭头指示的方向行驶",
            "指示汽车可以通过轨道的标志（如果指定为辅助标志，则仅是指定的汽车）",
            "前方有一个回旋处",
            "有一个岛形的安全区，供人们在有标志的道路上下车和行人过马路（禁止车辆进入）",
            "在此标志之前，车辆和缆车必须减速（减速是您可以立即停车的速度）",
            "车辆主要在道路交通中暂时停车",
            "由于路面不平，驾驶车辆时必须小心"
        ]

        kori_meaning = [
            "도로에 동물이 튀어 나올 우려가있다 (사슴 이외에도 원숭이와 여우, 곰 같은 것이있다)",
            "이 표지판이있는 곳에서는 차량이나 전차는 경음기 (경적)을 울리지 않으면 안",
            "이 먼저 환상 교차로가",
            "이 신호보다 먼저 보행자는 횡단 할 수없는",
            "차량 (자동차 · 경 자동차 · 원동기 장치 자전거)의 진입 못한다",
            "추월을 위해 오른쪽 부분 튀어 통행 금지",
            "차량은 화살표가 가리키는 방향으로 만 통행 할 수없는",
            "자동차가 궤도 담요 내를 통행 할 수있는 것을 나타내는 표지 (보조 표지 지정이있는 경우에는 그 지정 차량 만)",
            "앞으로 로타리가 있음을 나타내는",
            "표지판의 위치에 전차에 승강하는 사람이나 도로를 횡단하는 보행자를 위해 만들어진 섬 형상의 안전 지대가있는 것을 나타내는 표지 (차량 진입 불가)",
            "이 신호보다 먼저 차량 및 전차는 서행해야 말라 (서행는 즉시 정지 할 수있는 속도)",
            "주로 도로 교통에있어서 차량이 일시적으로 중지 할 수",
            "노면의 요철이 있기 때문에 차량의 운전에주의가 필요하다"
        ]


        
        return render_template('./result1.html', sing = predict_Confidence, japanese = japanese[predict_Confidence], meaning = meaning[predict_Confidence], eng_meaning = eng_meaning[predict_Confidence], chi_meaning = chi_meaning[predict_Confidence], kori_meaning = kori_meaning[predict_Confidence])




        



        # render_template('./result.html')
        #return render_template('./result.html', title='予想クラス', predict_Confidence=predict_Confidence)

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5000)