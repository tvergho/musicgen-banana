from potassium import Potassium, Request, Response
from audiocraft.models import MusicGen
import torch
from io import BytesIO
import soundfile as sf

app = Potassium("musicgen_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model.generate([prompt])
    
    # Assuming the outputs[0] is the audio data in numpy array format.
    audio_data = outputs[0].cpu().numpy()

    buf = BytesIO()
    sf.write(buf, audio_data, 16000, format='WAV')
    audio_bytes = buf.getvalue()

    return Response(
        json = {"audio": audio_bytes.decode('latin1'), "prompt": prompt},
        status=200
    )

if __name__ == "__main__":
    app.serve()