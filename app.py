import cv2

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open video
cap = cv2.VideoCapture(0) 

while True:
    # Read frame
    ret, frame = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display result    
    cv2.imshow('Video', frame)
    
    # Press q to exit video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
