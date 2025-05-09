from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json
import base64
import matplotlib.pyplot as plt
import io
from matplotlib.figure import Figure
import random
import requests

app = Flask(__name__)

# Configuraciones
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
PROCESSED_FOLDER = os.path.join(app.root_path, 'static/processed_media') # New folder for processed videos
MODEL_PATH = os.path.abspath(os.path.join(app.root_path, '../modelo_entrenado/modelo_celulas_entrenado_yolo_v8.pt'))

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True) # Create processed folder

# Cargar modelo YOLO
try:
    model = YOLO(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    print(f"Ruta del modelo: {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_media(): # Renamed from analyze_image
    if 'media' not in request.files: # Changed 'image' to 'media'
        return jsonify({'success': False, 'error': 'No se encontró ningún archivo'}), 400
    
    file = request.files['media'] # Changed 'image' to 'media'
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'}), 400
    
    original_filename = file.filename
    file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    
    # Generar nombre único para el archivo
    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}.{file_extension}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Procesar imagen con YOLO
            results = model(filepath)
            result = results[0]
            
            # Obtener imagen con detecciones
            img_with_boxes = result.plot() # This is a NumPy array
            
            # Convertir imagen a base64 para enviar directamente
            _, buffer = cv2.imencode(f'.{file_extension}', img_with_boxes)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Preparar datos de detecciones
            detections = []
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].item())
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    "id": i+1,
                    "class": class_name,
                    "confidence": round(confidence * 100, 2),
                    "bbox": [round(x, 2) for x in bbox]
                })
            
            # Eliminar archivo original subido después de procesar
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'success': True,
                'media_type': 'image',
                'image_base64': f'data:image/{file_extension};base64,{img_base64}',
                'detections': detections,
                'total_detections': len(detections)
            })

        elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            # Procesar video con YOLO
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'success': False, 'error': 'No se pudo abrir el video'}), 500

            # Obtener propiedades del video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Definir el codec y crear el objeto VideoWriter
            # Usar 'mp4v' para .mp4 o 'XVID' para .avi
            output_video_filename = f"processed_{unique_id}.mp4"
            output_video_path = os.path.join(PROCESSED_FOLDER, output_video_filename)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 - best browser compatibility
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    raise Exception("Failed to open VideoWriter with avc1 codec")
            except Exception as e:
                print(f"Error using avc1 codec: {e}")
                try:
                    # Fallback to MJPG in mp4 container
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    output_video_filename = f"processed_{unique_id}.avi"
                    output_video_path = os.path.join(PROCESSED_FOLDER, output_video_filename)
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                    if not out.isOpened():
                        raise Exception("Failed to open VideoWriter with MJPG codec")
                except Exception as e2:
                    print(f"Error using MJPG codec: {e2}")
                    # Last resort: mp4v
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video_filename = f"processed_{unique_id}.mp4"
                    output_video_path = os.path.join(PROCESSED_FOLDER, output_video_filename)
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
            frames_processed_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Realizar inferencia en el frame
                results = model(frame, stream=False) # stream=False para procesar frame a frame
                
                # Dibujar las detecciones en el frame
                processed_frame = results[0].plot() # results[0].plot() devuelve el frame con las cajas
                
                out.write(processed_frame)
                frames_processed_count += 1
            
            cap.release()
            out.release()
            
            # Eliminar archivo original subido después de procesar
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'success': True,
                'media_type': 'video',
                'video_url': f'/processed_media/{output_video_filename}',
                'frames_processed': frames_processed_count
            })
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': 'Formato de archivo no soportado'}), 400
            
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        # Asegurarse de eliminar el archivo subido en caso de error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed_media/<filename>') # New route for processed videos
def processed_media_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)