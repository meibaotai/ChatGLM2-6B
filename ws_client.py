import websocket
import _thread
import time
import rel
from transformers import AutoModel, AutoTokenizer
import requests

url = 'http://localhost:8080/generation/end'

def doPost(message):
    r = requests.post(url, message)
    pass

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model = model.eval()
    return tokenizer, model

def on_message(ws, message):
    prompt_text = message.prompt_text
    history = message.history
    past_key_values = message.past_key_values
    max_length = message.max_length
    top_p = message.top_p
    temperature = message.temperature
    for response, history, past_key_values in model.stream_chat(tokenizer, prompt_text, history,
                                                                past_key_values=past_key_values,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature,
                                                                return_past_key_values=True):
        ws.send(response)
    ws.send(history)
    ws.send(past_key_values)
    # doPost(data)
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

if __name__ == "__main__":
    ws_url = "ws://localhost:8080/websocket/localserver"
    tokenizer, model = get_model()
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(ws_url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)



    ws.run_forever(dispatcher=rel, reconnect=5)  # Set dispatcher to automatic reconnection, 5 second reconnect delay if connection closed unexpectedly
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()