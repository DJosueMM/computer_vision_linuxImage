import cv2

# Solicitar al usuario el nombre del archivo de video
video_file = input("Por favor, ingrese el nombre del archivo de video (incluya la extensión, por ejemplo, video.mp4): ")

# Abre el archivo de video
cap = cv2.VideoCapture(video_file)

# Obtiene el FPS del video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define el tamaño de la ventana para un monitor 1080p
window_width = 1920
window_height = 1080

# Crea una ventana con tamaño variable
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Define el tamaño de la ventana
cv2.resizeWindow('Frame', window_width, window_height)

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
    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
        break

# Libera los recursos y cierra la ventana
cap.release()
cv2.destroyAllWindows()
