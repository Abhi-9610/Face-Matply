import pickle
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import os
import uuid
from flask import Flask, request, jsonify

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

        # Check if either of the face encodings is None
        if registered_face_encoding is None or face_encoding is None:
            continue

        results = face_recognition.compare_faces([registered_face_encoding], face_encoding)
        if results[0]:
            return key

    return None


@app.route('/home')
def home():
    return "home"

@app.route('/check_face/<image_path>', methods=['POST'])
def check_face(image_path):
    raw_image_path = os.path.join('images', image_path)

    if allowed_file(raw_image_path):
        try:
            raw_picture = face_recognition.load_image_file(raw_image_path)
            raw_picture_rgb = cv2.cvtColor(raw_picture, cv2.COLOR_BGR2RGB)

            face_encodings = face_recognition.face_encodings(raw_picture_rgb)

         
            if face_encodings and len(face_encodings) > 0:
                raw_face_encoding = face_encodings[0]

                registered_id = is_registered(raw_face_encoding)
                user_details = ref.child(registered_id).get()

                if user_details:
                    user_name = user_details.get('name', 'User')
                    return jsonify({
                        'status': True,
                        'details': f"Welcome, {user_name}! You are registered with ID: {registered_id}"
                    }), 200
                else:
                    return jsonify({
                        'status': False,
                        'details': "You are not registered!!"
                    }), 404
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
    else:
        return jsonify({
            'status': False,
            'details': "Invalid image format"
        }), 400


@app.route('/registration', methods=['POST'])
def registration():
    uploaded_file = request.files.get('image', None)
    name = request.form.get('name')
    age = request.form.get('age')

    if not uploaded_file:
        return jsonify({'status': False, 'details': "Please provide an image"}), 400

    if not name:
        return jsonify({'status': False, 'details': "Please provide a name"}), 400

    if not age:
        return jsonify({'status': False, 'details': "Please provide an age"}), 400

    if allowed_file(uploaded_file.filename):
   
        filename_prefix = f"{name.lower().replace(' ', '_')}_{age}"
        unique_filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.jpg"

      
        firebase_safe_filename = unique_filename.replace('.', '_')

        local_image_path = os.path.join('images', unique_filename)
        uploaded_file.save(local_image_path)

   
        image = face_recognition.load_image_file(local_image_path)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            return jsonify({'status': False, 'details': "No face found in the provided image"}), 400

   
        encodings_directory = os.path.join('encodings')
        os.makedirs(encodings_directory, exist_ok=True)

    
        encodings_file_path = os.path.join(encodings_directory, f'{firebase_safe_filename}_encodings.p')
        with open(encodings_file_path, 'wb') as encodings_file:
            pickle.dump(face_encodings, encodings_file)

        storage_image_path = f"images/{firebase_safe_filename}"
        storage_blob = bucket.blob(storage_image_path)
        storage_blob.upload_from_filename(local_image_path)

        new_registration = {
            'name': name,
            'age': age,
            'encoding': face_encodings[0].tolist(),  # Save only the first face encoding
            'image_path': storage_image_path,
            'encodings_file_path': encodings_file_path
        }

      
        ref.child(firebase_safe_filename).set(new_registration)

        return jsonify({'status': True, "details": f"Registration successful! Your unique filename is: {unique_filename}"}), 201
    else:
        return jsonify({'status': False, 'details': "Invalid image format or no image provided"}), 400
if __name__ == '__main__':
    app.run(debug=True)
