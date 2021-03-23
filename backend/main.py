from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def greetings():
    return "greetings", 200

if __name__ == "__main__":
    app.run(debug=True)


# Ctrl +Alt + N starts running code
# Ctrl + Alt + M stops
# shortcuts are from Code Runner Extension