from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
from qalpr_v2.qalpr import process_ocr_codelab
from utils.bd_mongo_maestro import insert_or_update_record_maestro
from utils.bd_mongo_reportes import get_reportes_by_placa
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()



@app.post("/procesar-imagen/")
async def procesar_imagen(
    latitud: str,
    longitud: str,
    precision: float,
    file: UploadFile = File(...),
    coordenadas: str = "",
):
    """
    Endpoint que recibe una imagen, la procesa y registra en la base de datos maestro.

    Args:
        latitud (str): Latitud de la ubicación.
        longitud (str): Longitud de la ubicación.
        precision (float): Precisión de la geolocalización.
        file (UploadFile): Imagen cargada por el usuario.
        coordenadas (str): Coordenadas en formato "x1,y1,x2,y2" para recortar la imagen.

    Returns:
        JSONResponse: Resultado del procesamiento.
    """
    # Leer el contenido del archivo
    contenido = await file.read()
    try:
        imagen = Image.open(io.BytesIO(contenido))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida.",
        )

    # Recorte de la imagen según coordenadas
    if coordenadas:
        x1, y1, x2, y2 = map(float, coordenadas[1:-1].split(", "))
        imagen = imagen.crop((x1, y1, x2, y2))

    # Procesar la imagen con OCR
    resultado = process_ocr_codelab(imagen)

    # Consultar si la placa está reportada
    reportes = []
    placa = "N/A"
    reportada = False

    if len(resultado) >= 1:
        placa = resultado[0]["placa"].upper()
        reportes = get_reportes_by_placa(placa)
        if len(reportes) >= 1:
            reportada = True

            # Geolocalización
            geolocalizacion = {"latitud": latitud, "longitud": longitud}

            foto_ruta = f"fotos/{placa}.jpg"
            insert_or_update_record_maestro(
                placa=placa,
                foto=foto_ruta,
                geolocalizacion=geolocalizacion,
                resultado_listas=reportes
            )

            # Enviar correo
            enviar_correo(
                asunto="Alerta de placa reportada",
                destinatario="daniel.daza@quipux.com",
                mensaje=(
                    f"Se detectó una placa reportada: {placa}\n"
                    f"Ubicación: {geolocalizacion}\n"
                    f"Reportes: {reportes}"
                ),
            )

    # Respuesta final
    respuesta = {
        "placa": placa,
        "reportes": reportes,
        "reportada": reportada,
    }
    return respuesta


def enviar_correo(asunto: str, destinatario: str, mensaje: str):
    """
    Envía un correo electrónico utilizando el servidor SMTP.

    Args:
        asunto (str): Asunto del correo.
        destinatario (str): Dirección de correo del destinatario.
        mensaje (str): Cuerpo del correo.
    """
    remitente = "findplate2024@gmail.com"
    contraseña = "jvav tsfl ghct vmha"

    # Configurar el correo
    correo = MIMEMultipart()
    correo["From"] = remitente
    correo["To"] = destinatario
    correo["Subject"] = asunto
    correo.attach(MIMEText(mensaje, "plain"))

    # Conectar al servidor SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as servidor:
            servidor.starttls()
            servidor.login(remitente, contraseña)
            servidor.sendmail(remitente, destinatario, correo.as_string())
    except Exception as e:
        print(f"Error al enviar el correo: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)