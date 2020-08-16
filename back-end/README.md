# Final_year_project
Real Time Criminal Recognition

run "py face-recognizer.py" - training will be done and server will start
REST API created to access from web application
Real time recognition starts as soon as the server is up and will always be running 
http://127.0.0.1:5000/recognize-from-footage - to recognize from recorded footage
http://127.0.0.1:5000/recording?userid=1 - recording new face data and train
http://127.0.0.1:5000/record-from-footage?userid=1 - record face from footage of new criminal