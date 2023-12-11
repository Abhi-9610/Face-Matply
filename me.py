import pickle
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import os
from flask import Flask, request, jsonify
import base64
import numpy as np

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-system-ebb56-default-rtdb.firebaseio.com/",
    'storageBucket': "face-recognition-system-ebb56.appspot.com"
})
bucket = storage.bucket(app=firebase_admin.get_app())
ref = db.reference('/registered_faces')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_registered(face_encoding):
    registered_faces = ref.get() or {}
    for key, value in registered_faces.items():
        registered_face_encoding = value.get('encoding')

        if registered_face_encoding is None or face_encoding is None:
            continue

        results = face_recognition.compare_faces([registered_face_encoding], face_encoding)
        if results[0]:
            return key

    return None

def decode_base64_image(base64_string):
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def save_base64_image(base64_string, local_image_path):
    encoded_data = base64_string.split(',')[1]
    with open(local_image_path, 'wb') as f:
        f.write(base64.b64decode(encoded_data))

@app.route('/home')
def home():
    return "home"

@app.route('/check-face', methods=['POST'])
def check_or_register_face():
    try:
        # Retrieve data from the form
        base64_image = request.form.get('base64_image')
        name = request.form.get('name')
        age = request.form.get('age')
        if len(base64_image) > MAX_BASE64_LENGTH:
            return jsonify({'status': False, 'details': f"Base64 image length exceeds the allowed limit of {MAX_BASE64_LENGTH} bytes"}), 400
        if not base64_image:
            return jsonify({'status': False, 'details': "Please provide a base64-encoded image in the 'base64_image' field"}), 400

        raw_picture = decode_base64_image(base64_image)
        raw_picture_rgb = cv2.cvtColor(raw_picture, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(raw_picture_rgb)

        if face_encodings and len(face_encodings) > 0:
            raw_face_encoding = face_encodings[0]
            registered_id = is_registered(raw_face_encoding)

            if registered_id:
                user_details = ref.child(registered_id).get()
                user_name = user_details.get('name', 'User')
                return jsonify({
                    'status': True,
                    "message": "Welcome!!",
                    "name": user_name,
                    "registred_id": registered_id
                }), 200
            else:
                if not name or not age:
                    return jsonify({'status': False, 'details': "You are not Registred!!! Please Provide Name and Age"}), 400

                local_image_path = os.path.join('images', f"{name.lower().replace(' ', '_')}_{age}.jpg")
                save_base64_image(base64_image, local_image_path)

                image = face_recognition.load_image_file(local_image_path)
                face_encodings = face_recognition.face_encodings(image)

                if not face_encodings:
                    return jsonify({'status': False, 'details': "No face found in the provided image"}), 400

                encodings_directory = os.path.join('encodings')
                os.makedirs(encodings_directory, exist_ok=True)

                firebase_safe_filename = os.path.basename(local_image_path).replace('.', '_')
                encodings_file_path = os.path.join(encodings_directory, f'{firebase_safe_filename}_encodings.p')
                with open(encodings_file_path, 'wb') as encodings_file:
                    pickle.dump(face_encodings, encodings_file)

                storage_image_path = f"images/{firebase_safe_filename}"
                storage_blob = bucket.blob(storage_image_path)
                storage_blob.upload_from_filename(local_image_path)

                new_registration = {
                    'name': name,
                    'age': age,
                    'encoding': face_encodings[0].tolist(),
                    'image_path': storage_image_path,
                    'encodings_file_path': encodings_file_path
                }

                ref.child(firebase_safe_filename).set(new_registration)

                print("Location of the stored base64 image:", local_image_path)

                return jsonify({'status': True,
                                'details': f"Registration successful! Your unique filename is: {firebase_safe_filename}"}), 201

        else:
            return jsonify({
                'status': False,
                'details': "No face found in the provided image"
            }), 404

    except Exception as e:
        return jsonify({
            'status': False,
            'details': f"Error processing image: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
