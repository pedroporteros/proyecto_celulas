// Función para cargar las métricas del modelo
function loadModelMetrics() {
    fetch('/model-metrics')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error al cargar métricas:', data.error);
                return;
            }
            
            // Actualizar valores de resumen
            document.getElementById('mapValue').textContent = (data.map * 100).toFixed(1) + '%';
            document.getElementById('map50Value').textContent = (data.map50 * 100).toFixed(1) + '%';
            document.getElementById('precisionValue').textContent = (data.mean_precision * 100).toFixed(1) + '%';
            document.getElementById('recallValue').textContent = (data.mean_recall * 100).toFixed(1) + '%';
            
            // Generar gráficos
            createPrecisionRecallChart(data);
            createMAPChart(data);
            createLearningCurveChart(data);
            createConfusionMatrixChart(data);
            createMapEpochChart(data);
            
            // Mostrar la sección
            document.getElementById('performanceSection').classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Gráfico de Precisión y Recall por Clase
function createPrecisionRecallChart(data) {
    const ctx = document.getElementById('precisionRecallChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.classes.map(name => name.replace('_NORMAL', '')),
            datasets: [
                {
                    label: 'Precisión',
                    data: data.precision.map(p => p * 100),
                    backgroundColor: 'rgba(74, 111, 165, 0.7)',
                    borderColor: 'rgba(74, 111, 165, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Recall',
                    data: data.recall.map(r => r * 100),
                    backgroundColor: 'rgba(90, 185, 234, 0.7)',
                    borderColor: 'rgba(90, 185, 234, 1)',
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
                    title: {
                        display: true,
                        text: 'Porcentaje (%)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            plugins: {
                title: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.formattedValue + '%';
                        }
                    }
                }
            }
        }
    });
}

// Gráfico de mAP por umbral IoU
function createMAPChart(data) {
    const ctx = document.getElementById('mapChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.iou_thresholds.map(t => 'IoU ' + t),
            datasets: [{
                label: 'mAP',
                data: data.map_values.map(v => v * 100),
                fill: false,
                borderColor: 'rgba(76, 175, 80, 1)',
                backgroundColor: 'rgba(76, 175, 80, 0.2)',
                tension: 0.1,
                pointBackgroundColor: 'rgba(76, 175, 80, 1)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'mAP (%)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.formattedValue + '%';
                        }
                    }
                }
            }
        }
    });
}

// Gráfico de Curva de Aprendizaje
function createLearningCurveChart(data) {
    const ctx = document.getElementById('learningCurveChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Pérdida de Entrenamiento',
                    data: data.training_loss,
                    fill: false,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Pérdida de Validación',
                    data: data.validation_loss,
                    fill: false,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Épocas'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Pérdida'
                    }
                }
            }
        }
    });
}

// Gráfico de evolución de mAP por época
function createMapEpochChart(data) {
    const ctx = document.getElementById('mapEpochChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'mAP@0.5',
                    data: data.map50_by_epoch.map(v => v * 100),
                    fill: false,
                    borderColor: 'rgba(76, 175, 80, 1)',
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'mAP@0.5:0.95',
                    data: data.map50_95_by_epoch.map(v => v * 100),
                    fill: false,
                    borderColor: 'rgba(153, 102, 255, 1)',
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Épocas'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'mAP (%)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.formattedValue + '%';
                        }
                    }
                }
            }
        }
    });
}

// Gráfico de Matriz de Confusión
function createConfusionMatrixChart(data) {
    const ctx = document.getElementById('confusionMatrixChart').getContext('2d');
    
    // Transformar nombres largos a abreviaciones para mostrar mejor en el gráfico
    const shortClassNames = data.classes.map(name => {
        return name
            .replace('_NORMAL', '')
            .replace('NEUTROFILO_', 'NEUT_')
            .replace('LINFOCITO_', 'LINF_')
            .replace('ERITROCITO', 'ERITR')
            .replace('EOSINOFILO', 'EOSIN')
            .replace('TROMBOCITO', 'TROMB');
    });
    
    // Crear datasets para una matriz de confusión usando barras apiladas
    const datasets = [];
    for (let i = 0; i < data.conf_matrix.length; i++) {
        const backgroundColor = [];
        for (let j = 0; j < data.conf_matrix[i].length; j++) {
            // Color más intenso para valores diagonales (predicciones correctas)
            const intensity = i === j ? 0.9 : 
                              data.conf_matrix[i][j] / Math.max(...data.conf_matrix.flat());
            backgroundColor.push(`rgba(74, 111, 165, ${intensity})`);
        }
        
        datasets.push({
            label: shortClassNames[i] + ' (Real)',
            data: data.conf_matrix[i],
            backgroundColor: backgroundColor,
            borderColor: 'rgba(255, 255, 255, 0.5)',
            borderWidth: 1
        });
    }
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: shortClassNames,
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Predicción'
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Conteo'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label;
                            const value = context.raw;
                            return `Clase real: ${label}, Predicción: ${context.chart.data.labels[context.dataIndex]}, Valor: ${value}`;
                        }
                    }
                },
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 12
                    }
                },
                title: {
                    display: true,
                    text: 'Matriz de Confusión',
                    font: {
                        size: 14
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
    const resultImage = document.getElementById('resultImage');
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
            
            if (!selectedFile.type.match('image.*')) {
                alert('Por favor, selecciona una imagen válida.');
                return;
            }
            
            // Previsualizar la imagen
            const reader = new FileReader();
            reader.onload = function(e) {
                originalImage.src = e.target.result;
                // Habilitar botón de análisis
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(selectedFile);
        }
    }

    // Manejar análisis de imagen
    analyzeBtn.addEventListener('click', analyzeImage);

    function analyzeImage() {
        if (!selectedFile) {
            alert('Por favor, selecciona una imagen primero.');
            return;
        }

        // Mostrar carga
        loadingOverlay.classList.remove('hidden');
        
        // Preparar datos para enviar
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Enviar solicitud al servidor
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Mostrar imagen resultante
                resultImage.src = data.image_base64;
                
                // Limpiar y llenar tabla de detecciones
                detectionsTable.innerHTML = '';
                data.detections.forEach(detection => {
                    const row = detectionsTable.insertRow();
                    
                    const idCell = row.insertCell(0);
                    const classCell = row.insertCell(1);
                    const confidenceCell = row.insertCell(2);
                    const bboxCell = row.insertCell(3);
                    
                    idCell.textContent = detection.id;
                    classCell.textContent = detection.class;
                    confidenceCell.textContent = `${detection.confidence.toFixed(2)}%`;
                    bboxCell.textContent = `[${detection.bbox.join(', ')}]`;
                });
                
                // Actualizar contador
                detectionCount.textContent = data.total_detections;
                
                // Mostrar resultados
                resultsContainer.classList.remove('hidden');
                
                // Desplazarse a los resultados
                window.scrollTo({
                    top: resultsContainer.offsetTop,
                    behavior: 'smooth'
                });
            } else {
                alert('Error al analizar la imagen: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Ocurrió un error al procesar la imagen');
        })
        .finally(() => {
            // Ocultar carga
            loadingOverlay.classList.add('hidden');
        });
    }

    // Cargar las métricas del modelo
    loadModelMetrics();
});