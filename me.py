import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import os
import uuid
from flask import Flask, request, render_template

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
@app.route('/check_face/<image_path>', methods=['GET', 'POST'])
def check_face(image_path):
    # Combine the provided image path with the 'images/' folder
    raw_image_path = os.path.join('images', image_path)

    if allowed_file(raw_image_path):
        try:
            # Load the image in RGB format
            raw_picture = face_recognition.load_image_file(raw_image_path)
            raw_picture_rgb = cv2.cvtColor(raw_picture, cv2.COLOR_BGR2RGB)

            # Get the face encoding
            raw_face_encoding = face_recognition.face_encodings(raw_picture_rgb)[0]

            # Checking if the face is registered
            registered_id = is_registered(raw_face_encoding)
            user_details = ref.child(registered_id).get()

            if user_details:
                user_name = user_details.get('name', 'User')
                return f"Welcome, {user_name}! You are registered with ID: {registered_id}"
            else:
                # Render the registration form with the face encoding
                return f"You are not registered!!"

        except Exception as e:
            return f"Error processing image: {str(e)}"
    else:
        return "Invalid format!!"


# Route for displaying Registration from Front-End
@app.route('/registration',methods=['GET', 'POST'])
def registration():
    unique_id = str(uuid.uuid4())

    # Upload the raw image to Firebase Storage
    raw_image_blob = bucket.blob(f"images/{unique_id}.jpg")

    # Use get method with a default value of None
    uploaded_file = request.files.get('image', None)

    if uploaded_file:
        # Upload the file to Firebase Storage
        raw_image_blob.upload_from_file(uploaded_file)

        # Store the details and encoding in the Firebase Realtime Database from the backend
        new_registration = {
            'name': request.form.get('name', ''),
            'age': request.form.get('age', ''),
            'encoding': list(map(float, request.form.getlist('encoding[]', []))),
            'image_path': f"images/{unique_id}.jpg"
        }

        ref.child(unique_id).set(new_registration)

        return f"Registration successful! Your unique ID is: {unique_id}"
    else:
        return "No image file provided in the request."

if __name__ == '__main__':
    app.run(debug=True)
