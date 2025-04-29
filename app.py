from flask import Flask, render_template, request, jsonify
from dialog_flow_agent import dialogue_mgmt_system as data_analysis_chat
from dialog_flow import dialogue_mgmt_system as laptop_chat
from dialog_flow_restaurant import dialogue_mgmt_system as restaurant_chat
from insurance_dialog_flow import insurance_dialog_flow
import queue
import threading

app = Flask(__name__)

# Store active conversations
conversations = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat/<system_type>")
def chat(system_type):
    return render_template("chatbot.html", system_type=system_type)

@app.route("/api/chat/<system_type>", methods=["POST"])
def handle_chat(system_type):
    data = request.json
    user_input = data.get("message")
    conversation_id = data.get("conversation_id")
    
    if conversation_id not in conversations:
        # Initialize new conversation
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        conversations[conversation_id] = {
            "input_queue": input_queue,
            "output_queue": output_queue,
            "thread": None
        }
        
        def run_dialogue_system():
            if system_type == "data_analysis":
                data_analysis_chat(conversations[conversation_id])
            elif system_type == "restaurant":
                restaurant_chat(conversations[conversation_id])
            elif system_type == "insurance":
                insurance_dialog_flow(conversations[conversation_id])
        
        thread = threading.Thread(target=run_dialogue_system)
        thread.start()
        conversations[conversation_id]["thread"] = thread
    
    # Put user input in queue
    conversations[conversation_id]["input_queue"].put(user_input)
    
    try:
        # Get response from dialogue system
        response = conversations[conversation_id]["output_queue"].get(timeout=60)
        return jsonify({"response": response})
    except queue.Empty:
        return jsonify({"response": "Sorry, the system took too long to respond. Please try again."})

if __name__ == "__main__":
    app.run(debug=True)

  