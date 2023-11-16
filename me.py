
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import os
import uuid
from flask import Flask, request, render_template

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-system-ebb56-default-rtdb.firebaseio.com/",
    'storageBucket': "face-recognition-system-ebb56.appspot.com"
})
bucket = storage.bucket(app=firebase_admin.get_app())
ref = db.reference('/registered_faces')

# Function to check if a face is registered
def is_registered(face_encoding):
    registered_faces = ref.get()
    if registered_faces is not None:
        for key, value in registered_faces.items():
            registered_face_encoding = value['encoding']
            results = face_recognition.compare_faces([registered_face_encoding], face_encoding)
            if results[0]:
                return key  # Return the unique ID if a match is found
    return None  # Return None if no match is found

# Route for home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# Route for registration form
@app.route('/check_face', methods=['GET', 'POST'])
def check_face():
    
        # Manually provide the image path for testing and Raw image should be Provided by backend
        raw_image_path = "images/me.jpg"

        # we are uploading image for face recognition
        raw_picture = face_recognition.load_image_file(raw_image_path)
        raw_face_encoding = face_recognition.face_encodings(raw_picture)[0]

        # Checking if the face is registered
        registered_id = is_registered(raw_face_encoding)

        if registered_id is not None:
            return f"Welcome! You are registered with ID: {registered_id}"
        else:
            # Render the registration form with the face encoding
            return  f"You are not registred!!"

   

# Route for displaying Registration from Front-End
@app.route('/registration')
def registration():
    unique_id = str(uuid.uuid4())

    # Upload the raw image to Firebase Storage
    raw_image_path_storage = f"images/{unique_id}.jpg"
    raw_image_blob = bucket.blob(raw_image_path_storage)
    raw_image_blob.upload_from_file(request.files['image'])

    # Store the details and encoding in the Firebase Realtime Database from the backend
    new_registration = {
        'name': request.form['name'],
        'age': request.form['age'],
        'encoding': list(map(float, request.form.getlist('encoding[]'))),
        'image_path': raw_image_path_storage
    }

    ref.child(unique_id).set(new_registration)

    return f"Registration successful! Your unique ID is: {unique_id}"

if __name__ == '__main__':
    app.run(debug=True)
