import cv2

# Solicitar al usuario el nombre del archivo de video
video_file = input("Por favor, ingrese el nombre del archivo de video (incluya la extensión, por ejemplo, video.mp4): ")

# Inicializar la fuente de vídeo
cap = cv2.VideoCapture(video_file)

# Verificar si la captura de vídeo se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la fuente de video.")
    exit()

# Obtener el FPS del video fuente
fps = cap.get(cv2.CAP_PROP_FPS)

#PIPELINE
while True:
    # Leer un fotograma de la fuente de vídeo
    ret, frame = cap.read()

    # Verificar si el fotograma se leyó correctamente
    if not ret:
        print("El video finalizó.")
        break

    # Mostrar el fotograma procesado en una ventana
    cv2.imshow('Frame', frame)

    # Esperar el tiempo necesario para mantener el FPS original
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
