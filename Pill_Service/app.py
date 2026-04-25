import os
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import logging

app = Flask(__name__)
# 모든 도메인 허용 (개발 단계용, 배포 시 특정 도메인만 지정 가능)
CORS(app)

# 1. 로깅 설정 (누가 언제 분석했는지 기록)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. 모델 및 폴더 설정
MODEL_PATH = "best.pt"
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

try:
    model = YOLO(MODEL_PATH)
    logger.info("✅ YOLO 모델 로드 완료")
except Exception as e:
    logger.error(f"❌ 모델 로드 실패: {e}")

# 3. 임시 데이터베이스 (현업에선 MySQL/Firebase 등을 쓰지만, 우선 메모리에 저장)
# 하드웨어가 이 정보를 읽어갈 겁니다.
pill_status = {
    "is_taken": False,
    "last_pill_name": "",
    "taken_at": ""
}

@app.route('/predict', methods=['POST'])
def predict():
    """프론트엔드에서 사진을 받아 분석하고 결과를 저장하는 API"""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "이미지 파일이 전송되지 않았습니다."}), 400

    file = request.files['image']

    # 파일명에 시간을 붙여 중복 방지
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # 모델 예측
        results = model.predict(source=file_path, conf=0.5) # 정확도 50% 이상만 추출

        detected_pills = []
        for r in results:
            for box in r.boxes:
                name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                detected_pills.append({
                    "pill_name": name,
                    "confidence": f"{conf*100:.1f}%"
                })

        # 분석 성공 시 상태 업데이트 (하드웨어 연동용)
        if detected_pills:
            pill_status["is_taken"] = True
            pill_status["last_pill_name"] = detected_pills[0]["pill_name"]
            pill_status["taken_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "status": "success",
            "timestamp": pill_status["taken_at"],
            "data": detected_pills,
            "image_url": f"/static/uploads/{filename}" # 프론트에서 확인할 수 있게 경로 반환
        }), 200

    except Exception as e:
        logger.error(f"분석 중 에러 발생: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/hardware/status', methods=['GET'])
def get_hardware_status():
    """ESP32 하드웨어가 약 복용 여부를 확인하기 위해 호출하는 API"""
    return jsonify(pill_status), 200

@app.route('/hardware/reset', methods=['POST'])
def reset_status():
    """다음 복용 시간을 위해 상태를 초기화 (하드웨어 버튼 등으로 호출 가능)"""
    pill_status["is_taken"] = False
    return jsonify({"status": "reset_success"}), 200

if __name__ == '__main__':
    # 0.0.0.0으로 열어야 같은 와이파이의 스마트폰(프론트)과 ESP32가 접속 가능합니다.
    app.run(host='0.0.0.0', port=5000, debug=False)