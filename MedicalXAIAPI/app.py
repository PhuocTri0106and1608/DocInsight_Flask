from flask import Flask, request, jsonify, url_for
from model import shap_model
from flask_cors import CORS, cross_origin
from PIL import Image
import cloudinary
from cloudinary.uploader import upload
import os
import io

app = Flask(__name__)
CORS(app)
    
cloudinary.config(cloud_name=os.getenv('CLOUDINARY_NAME'), api_key=os.getenv('API_KEY'), 
    api_secret=os.getenv('API_SECRET'))

@app.route("/XAI", methods=['POST'])
@cross_origin()
def XAI():
    if 'upload' not in request.files:
        return jsonify({'error': 'No upload part'})
    f = request.files['upload']
    if f.filename == "":
        return jsonify({'error': 'No selected file'})
    
    try:
        with Image.open(f) as img:
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            input_url = upload(buffer)
        
        prediction = shap_model(f)
        result_path = "result.png"
        with Image.open(result_path) as res:
            buffer_result = io.BytesIO()
            res.save(buffer_result, format='PNG')
            buffer_result.seek(0)
            result_url = upload(buffer_result)
        
        return jsonify({
            'input_url': input_url['secure_url'],
            'result_url': result_url['secure_url'],
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
def create_app():
    return app
