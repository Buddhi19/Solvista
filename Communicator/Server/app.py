from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import time
import json
import os
from main import process_image, calculate_expression, process_image_for_whiteboard, save_bode_plot
from main import fourier_solver, fourier_transform_image
from main import laplace_solver, laplace_equation_image, laplace_spectrum_image
from main import calculate_exp
from main import plot_graph, get_num_and_den
from main import get_z_transform
from main import solve_for_x

app = FastAPI()

# host_url = '192.168.1.5'
# host_url = '10.30.1.107'
# host_url = '192.168.8.103'
host_url = '192.168.43.237'

@app.get("/")
def read_root():
    return 

@app.post("/image")
async def image_route(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    file_path = "img.png"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    result = process_image(file_path)

    # Clean up the file after processing
    return {"result": result}

@app.post("/image_whiteboard")
async def image_route_whiteboard(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    file_path = "img.png"
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    result = process_image_for_whiteboard(file_path)

    # Clean up the file after processing
    os.remove(file_path)

    return {"result": result}

@app.post("/generate_bode_plot")
async def generate_bode_plot(data: dict):
    numerator = data.get('numerator')
    denominator = data.get('denominator')
    path = save_bode_plot(numerator, denominator)  # Call your function to generate and save the Bode plot
    return FileResponse(path, media_type='image/png')

@app.post("/fourier_transform_image")
async def fourier_transform(data: dict):
    expression = data.get('expression')
    a = data.get('a')
    b = data.get('b')
    fourier = fourier_solver(expression, a, b)
    path = fourier_transform_image()
    return FileResponse(path, media_type='image/png')

@app.post("/lap_transform_image")
async def laplace_transform(data: dict):
    expression = data.get('expression')
    a = data.get('a')
    b = data.get('b')
    laplace = laplace_solver(expression, a, b)
    path = laplace_equation_image()
    return FileResponse(path, media_type='image/png')

@app.post("/laplace_spectrum_image")
async def laplace_spectrum(data : dict):
    exp = data.get('expression')
    path = laplace_spectrum_image()
    return FileResponse(path, media_type='image/png')

@app.post("/calculate")
async def calculate(data: dict):
    expression = data.get('expression')
    if not expression:
        return {"result": []}
    ans = calculate_exp(expression)
    return {"result": ans}

@app.post("/plot_graph")
async def send_plot(data: dict):
    expression = data.get('expression')
    if not expression:
        return {"result": []}
    path = plot_graph(expression)
    return FileResponse(path, media_type='image/png') 

@app.post("/transfer_function")
async def transfer_function(data: dict):
    expression = data.get('expression')
    if not expression:
        return {"result": []}
    numerator, denominator = get_num_and_den(expression)
    return {"result": "pass", "numerator": str(numerator), "denominator" : str(denominator)}
    
@app.post("/z_transform_image")
async def z_transform_image(data: dict):
    expression = data.get('expression')
    path = get_z_transform(expression)
    return FileResponse(path, media_type='image/png')

@app.post("/solve_for_x")
async def solving_for_x(data: dict):
    expression = data.get('expression')
    if not expression:
        return {"result": "No expression provided"}
    ans = solve_for_x(expression)
    return {"result": ans}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=host_url, port=80)
