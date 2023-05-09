'''
This file is a barebones FastAPI example that:
  1. Accepts GET request, renders a HTML form at localhost:8000 allowing the user to
     upload a image and select YOLO model, then submit that data via POST
  2. Accept POST request, run YOLO model on input image, return JSON output

Works with client_minimal.py

This script does not require any of the HTML templates in /templates or other code in this repo
and does not involve stuff like Bootstrap, Javascript, JQuery, etc.
'''

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse

from PIL import Image
from io import BytesIO

import pytesseract
import numpy as np
import torch

app = FastAPI()

@app.get("/")
async def home(request: Request):
  ''' Returns barebones HTML form allowing the user to select a file and model '''

  html_content = '''
<form method="post" enctype="multipart/form-data">
  <div>
    <label>Upload Image</label>
    <input name="file" type="file" multiple>
    <div>
      <label>Select YOLO Model</label>
      <select name="model_name">
        <option>yolov5s</option>
      </select>
    </div>
  </div>
  <button type="submit">Submit</button>
</form>
'''

  return HTMLResponse(content=html_content, status_code=200)


@app.post("/")
async def process_home_form(file: UploadFile = File(...), model_name: str = Form(...)):
  
    '''
    Requires an image file upload, model name (ex. yolov5s).
    Returns: json response with list of list of dicts.
      Each dict contains class, class_name, confidence, normalized_bbox

    Note: Because this is an async method, the YOLO inference is a blocking
    operation.
    '''
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload = False)
    img = Image.open(BytesIO(await file.read()))
    #This is how you decode + process image with PIL
    results = model(img)
    #This is how you decode + process image with OpenCV + numpy
    #results = model(cv2.cvtColor(cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))

    json_results = results_to_json(results,model,img)
    return json_results


def results_to_json(results, model,img):
    ''' Helper function for process_home_form()'''
    json_results = []
    for result in results.xyxy:
        json_result = []
        for pred in result:
            class_id = int(pred[5])
            class_name = model.model.names[class_id]
            bbox = [int(x) for x in pred[:4].tolist()]
            confidence = float(pred[4])

            if class_name == 'license_plate':   
                x1, y1, x2, y2 = bbox
                img_crop = img.crop((x1, y1, x2, y2))
                # Apply OCR to the license plate region
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                ocr_result = pytesseract.image_to_string(img_crop, config='--psm 6')
                # Remove any leading/trailing whitespace from the OCR result
                ocr_result = ocr_result.strip()         

                pred_dict = {
                    "class": class_id,
                    "class_name": class_name,
                    "bbox": bbox,
                    "confidence": confidence,
                    "ocr_result": ocr_result# Remove any leading/trailing whitespace from the OCR result
                }
            else:
                # Add dict without license plate number
                pred_dict = {
                    "class": class_id,
                    "class_name": class_name,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            json_result.append(pred_dict)

        json_results.append(json_result)

    return json_results

if __name__ == '__main__':
    import uvicorn
    
    app_str = 'server_minimal:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)