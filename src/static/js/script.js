// Función para cargar las métricas del modelo
function loadModelMetrics() {
    fetch('https://script.google.com/macros/s/AKfycbwlxm-mSFqnD2Jkhc9BwOMfrKptfxiDZDtpYlX4Wp5HiKtM3a3CAuyTXsATICoW-wP2/exec')
        .then(response => response.json())
        .then(data => {
            // ...el resto de tu lógica permanece igual...
            // 1) Elegir el modelo (la clave “YYYY_MM_DD_modelo” más reciente)
            const modelKeys = Object.keys(data);
            if (modelKeys.length === 0) {
                console.error('No hay modelos en la respuesta');
                return;
            }
            modelKeys.sort();
            const latestKey = modelKeys[modelKeys.length - 1];
            const metrics   = data[latestKey];

            // 2) Actualizar valores de summary
            const s = metrics.summary;
            document.getElementById('mapValue').textContent =
              (s.map50_95 * 100).toFixed(1) + '%';
            document.getElementById('map50Value').textContent =
              (s.map50 * 100).toFixed(1) + '%';
            document.getElementById('precisionValue').textContent =
              (s.mean_precision * 100).toFixed(1) + '%';
            document.getElementById('recallValue').textContent =
              (s.mean_recall * 100).toFixed(1) + '%';

            const f1El = document.getElementById('f1Value');
            if (f1El) {
              f1El.textContent = (s.mean_f1_score * 100).toFixed(1) + '%';
            }
            const finalMapEl = document.getElementById('finalMapValue');
            if (finalMapEl) {
              finalMapEl.textContent =
                (s.final_map_iou_0_95 * 100).toFixed(1) + '%';
            }

            // 3) Generar gráficos con el objeto `metrics`
            createPrecisionRecallChart(metrics);
            createMapEpochChart(metrics);
            createLearningCurveChart(metrics);

            document.getElementById('performanceSection')
                    .classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error al cargar métricas:', error);
        });
}


// Precisión vs Recall por clase
function createPrecisionRecallChart(metrics) {
    const ctx = document.getElementById('precisionRecallChart')
                         .getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: metrics.class_metrics
                            .map(c => c.class.replace('_NORMAL', '')),
            datasets: [
                {
                    label: 'Precisión',
                    data: metrics.class_metrics
                                 .map(c => c.precision * 100),
                    borderWidth: 1
                },
                {
                    label: 'Recall',
                    data: metrics.class_metrics
                                 .map(c => c.recall * 100),
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Porcentaje (%)' }
                },
                x: { ticks: { maxRotation: 45, minRotation: 45 } }
            },
            plugins: {
                tooltip: {
                  callbacks: {
                    label: ctx => `${ctx.dataset.label}: ${ctx.formattedValue}%`
                  }
                }
            }
        }
    });
}


// Curva de aprendizaje (loss)
function createLearningCurveChart(metrics) {
    const lc = metrics.learning_curve;
    const ctx = document.getElementById('learningCurveChart')
                         .getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: lc.epoch,
            datasets: [
                {
                    label: 'Pérdida Entrenamiento',
                    data: lc.training_loss,
                    fill: false,
                    tension: 0.1,
                    borderWidth: 2
                },
                {
                    label: 'Pérdida Validación',
                    data: lc.validation_loss,
                    fill: false,
                    tension: 0.1,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Épocas' }},
                y: { title: { display: true, text: 'Pérdida' }}
            }
        }
    });
}


// Evolución de mAP por época
function createMapEpochChart(metrics) {
    const lc = metrics.learning_curve;
    const ctx = document.getElementById('mapEpochChart')
                         .getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: lc.epoch,
            datasets: [
                {
                    label: 'mAP@0.5',
                    data: lc.map50.map(v => v * 100),
                    fill: false,
                    tension: 0.1,
                    borderWidth: 2
                },
                {
                    label: 'mAP@0.5:0.95',
                    data: lc['map50-95'].map(v => v * 100),
                    fill: false,
                    tension: 0.1,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Épocas' }},
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'mAP (%)' }
                }
            },
            plugins: {
                tooltip: {
                  callbacks: {
                    label: ctx => `${ctx.dataset.label}: ${ctx.formattedValue}%`
                  }
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Elementos del DOM
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const originalImage = document.getElementById('originalImage');
    const originalVideoPreview = document.getElementById('originalVideoPreview'); // New
    const resultImage = document.getElementById('resultImage');
    const resultVideo = document.getElementById('resultVideo'); // New
    const detectionsTable = document.getElementById('detectionsTable').getElementsByTagName('tbody')[0];
    const detectionCount = document.getElementById('detectionCount');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Asegurarnos de que el overlay esté oculto inicialmente
    loadingOverlay.classList.add('hidden');
    
    let selectedFile = null;

    // Eventos para arrastrar y soltar
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('active');
    }

    function unhighlight() {
        dropArea.classList.remove('active');
    }

    // Manejar el archivo soltado
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Manejar click en el área de upload
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Manejar selección de archivo
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length) {
            selectedFile = files[0];
            
            // Ocultar vistas previas anteriores y limpiar sources
            originalImage.style.display = 'none';
            originalImage.src = '#';
            if (originalVideoPreview) {
                originalVideoPreview.style.display = 'none';
                URL.revokeObjectURL(originalVideoPreview.src); // Limpiar src anterior si era un object URL
                originalVideoPreview.src = '';
            }

            if (selectedFile.type.match('image.*')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    originalImage.src = e.target.result;
                    originalImage.style.display = 'block';
                };
                reader.readAsDataURL(selectedFile);
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analizar Imagen';
            } else if (selectedFile.type.match('video.*')) {
                if (originalVideoPreview) {
                    originalVideoPreview.src = URL.createObjectURL(selectedFile);
                    originalVideoPreview.style.display = 'block';
                } else { 
                    // Fallback si el elemento no existe
                    originalImage.alt = 'Vista previa de video no disponible';
                    originalImage.style.display = 'block'; // Show alt text in image tag
                }
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analizar Video';
            } else {
                alert('Por favor, selecciona una imagen o video válido.');
                selectedFile = null;
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = 'Analizar Archivo';
                return;
            }
            // Ocultar resultados anteriores al cargar nuevo archivo
            resultsContainer.classList.add('hidden');
            resultImage.style.display = 'none';
            if (resultVideo) resultVideo.style.display = 'none';

        }
    }

    // Manejar análisis de archivo
    analyzeBtn.addEventListener('click', analyzeMedia); // Renamed

    function analyzeMedia() { // Renamed from analyzeImage
        if (!selectedFile) {
            alert('Por favor, selecciona un archivo primero.');
            return;
        }

        // Mostrar carga y actualizar texto según el tipo de archivo
        loadingOverlay.classList.remove('hidden');
        const loadingText = loadingOverlay.querySelector('p');
        if (selectedFile.type.match('video.*')) {
            loadingText.textContent = 'Analizando video... Esto puede tardar varios minutos.';
        } else {
            loadingText.textContent = 'Analizando imagen...';
        }
        
        // Preparar datos para enviar
        const formData = new FormData();
        formData.append('media', selectedFile); // Changed 'image' to 'media' for clarity
        
        // Enviar solicitud al servidor
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Ocultar vistas previas de resultados anteriores
                resultImage.style.display = 'none';
                resultImage.src = '#';
                if (resultVideo) {
                    resultVideo.style.display = 'none';
                    resultVideo.src = '';
                }

                if (data.media_type === 'image') {
                    resultImage.src = data.image_base64;
                    resultImage.style.display = 'block';
                    
                    detectionsTable.innerHTML = ''; // Limpiar tabla
                    if (data.detections && data.detections.length > 0) {
                        data.detections.forEach(detection => {
                            const row = detectionsTable.insertRow();
                            row.insertCell(0).textContent = detection.id;
                            row.insertCell(1).textContent = detection.class;
                            row.insertCell(2).textContent = `${detection.confidence.toFixed(2)}%`;
                            row.insertCell(3).textContent = `[${detection.bbox.join(', ')}]`;
                        });
                    } else {
                        detectionsTable.innerHTML = '<tr><td colspan="4">No se encontraron detecciones.</td></tr>';
                    }
                    detectionCount.textContent = data.total_detections !== undefined ? data.total_detections : 0;
                } else if (data.media_type === 'video') {
                    if (resultVideo) {
                        resultVideo.src = data.video_url; // Server will provide URL to processed video
                        resultVideo.load();
                        resultVideo.style.display = 'block';
                    }
                    detectionsTable.innerHTML = '<tr><td colspan="4">Análisis de video completado. Las detecciones están en el video.</td></tr>';
                    detectionCount.textContent = data.frames_processed !== undefined ? `${data.frames_processed} frames` : 'N/A';
                }
                
                resultsContainer.classList.remove('hidden');
                window.scrollTo({ top: resultsContainer.offsetTop, behavior: 'smooth' });
            } else {
                alert('Error al analizar el archivo: ' + (data.error || 'Error desconocido'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Ocurrió un error al procesar el archivo.');
        })
        .finally(() => {
            loadingOverlay.classList.add('hidden');
            loadingText.textContent = 'Analizando archivo...'; // Reset loading text
        });
    }

    // Cargar las métricas del modelo
    loadModelMetrics();
});