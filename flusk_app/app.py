from flask import Flask, request, jsonify
from dotenv import load_dotenv

from document_detector import DocumentDetector
from flusk_app.boundingbox.boundingbox import crop_nic_new_bounding_box, \
    crop_driving_licence_bounding_box, crop_nic_old_bounding_box
from flusk_app.image_classifer.id_classifier import IDClassifier
from flusk_app.ocr.doc_ocr import extract_text

# Load environment variables from .env if present (optional)
load_dotenv()


def create_app():
    app = Flask(__name__)

    @app.post("/api/verify_image")
    def verify_image():
        data = request.get_json(silent=True) or {}

        img_data = data.get("img_data")

        try:
            class_names = ['driving_licence', 'new_nic', 'old_nic', 'passport']
            model_path = r'models/image_classifier_fine_tuned.h5'

            classifier = IDClassifier(
                model_or_path=model_path,
                class_names=class_names,
                dense_to_patch=None,
                auto_patch=True
            )

            result = classifier.predict(img_data, confidence_threshold=0.86)
            class_name = result['class_name']

            cropped_image = None

            if class_name == "nic_old":
                cropped_image = crop_nic_old_bounding_box()
            elif class_name == "nic_new":
                cropped_image = crop_nic_new_bounding_box()
            elif class_name == "driving_licence":
                cropped_image = crop_driving_licence_bounding_box()

            res = extract_text(cropped_image)
            return jsonify(res), 200
        except Exception as e1:
            try:
                res = extract_text(img_data)
                return jsonify(res), 200
            except Exception as e2:
                return jsonify({"error": "Unable to process image"}), 400

    return app


app = create_app()

if __name__ == "__main__":
    # Default dev server: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
