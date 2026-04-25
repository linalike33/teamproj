from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# 팀원이 준 AI 모델 파일 불러오기
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. 앱에서 보낸 사진 파일 받기
    if 'file' not in request.files:
        return jsonify({"status": "fail", "message": "No file uploaded"})

    file = request.files['file']
    img_path = "temp_image.jpg"
    file.save(img_path)

    # 2. AI 모델로 알약 분석하기
    results = model(img_path)

    # 3. 결과 분석 및 응답 전송
    if len(results[0].boxes) > 0:
        # 가장 확률이 높은 알약의 번호와 이름 가져오기
        top_result = results[0].boxes[0]
        class_id = int(top_result.cls)
        pill_name = model.names[class_id] # yaml 파일에 있던 이름들

        return jsonify({
            "status": "success",
            "pillName": pill_name
        })
    else:
        return jsonify({
            "status": "fail",
            "message": "등록되지 않은 알약입니다."
        })

if __name__ == '__main__':
    # 5000번 포트로 서버 실행
    app.run(host='0.0.0.0', port=5000)