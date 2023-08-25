from fastapi import FastAPI
from paint_new import paintScreen
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(paintScreen(), media_type="multipart/x-mixed-replace;boundary=frame")
