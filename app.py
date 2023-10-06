from potassium import Potassium, Request, Response
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
from io import BytesIO
import soundfile as sf

app = Potassium("musicgen_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium", torch_dtype=torch.float16)
   
    context = {
        "model": model,
        "processor": processor
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    processor = context.get("processor")
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, max_new_tokens=256)
    
    # Assuming the outputs[0] is the audio data in numpy array format.
    audio_data = audio_values[0].cpu().numpy().astype("float32")

    buf = BytesIO()
    sf.write(buf, audio_data, 32000, format='WAV')
    audio_bytes = buf.getvalue()

    return Response(
        json = {"audio": audio_bytes.decode('latin1'), "prompt": prompt},
        status=200
    )

if __name__ == "__main__":
    app.serve()