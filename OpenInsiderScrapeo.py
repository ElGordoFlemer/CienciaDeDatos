import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import os
import time  # Para el retraso entre páginas

d = 60  # cantidad de dias
m = 500   # monto

# URL base del screener de OpenInsider (sin el parámetro page)
base_url = f"http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd={d}&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl={m}&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page="

headers_request = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Lista para acumular todas las filas de todas las páginas
all_rows = []
page = 1
headers = None  # Se obtendrán en la primera página

print("Iniciando descarga de datos de OpenInsider...")

while True:
    # Construir URL para la página actual
    url = base_url + str(page)
    
    try:
        response = requests.get(url, headers=headers_request, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar la página {page}: {e}")
        break

    soup = BeautifulSoup(response.text, "html.parser")

    # Buscar la tabla principal
    table = soup.find("table", class_="tinytable")

    if not table:
        print(f"No se encontró tabla en la página {page}. Deteniendo.")
        break

    # Obtener encabezados solo en la primera página
    if headers is None:
        headers = [th.text.strip() for th in table.find_all("th")]
        print(f"Encabezados encontrados: {len(headers)} columnas")

    # Obtener filas de datos de esta página
    current_rows = []
    for tr in table.find_all("tr")[1:]:  # Omitir la fila de encabezado
        cells = [td.text.strip() for td in tr.find_all("td")]
        
        # Asegurar que el número de celdas sea consistente (como en el original)
        if len(cells) < len(headers):
            cells.extend([''] * (len(headers) - len(cells)))
        
        cells = cells[:len(headers)]
        
        if len(cells) == len(headers):
            # Crear un diccionario para cada fila
            row_data = dict(zip(headers, cells))
            current_rows.append(row_data)

    # Si no hay filas en esta página, detener
    if len(current_rows) == 0:
        print(f"No hay más datos en la página {page}. Total descargado: {len(all_rows)} filas.")
        break

    # Agregar las filas actuales al total
    all_rows.extend(current_rows)
    print(f"Página {page}: descargadas {len(current_rows)} filas. Total acumulado: {len(all_rows)}")

    # Incrementar página y esperar un segundo
    page += 1
    time.sleep(1)

# Si no hay datos, salir
if not all_rows:
    print("No se encontraron datos para descargar.")
    exit()

# Generar nombre del archivo con fecha (ddmmaaaa)
fecha_hoy = datetime.now().strftime("%d%m%Y")
filename = f"insider_purchases_{fecha_hoy}.csv"

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, filename)

# Guardar archivo CSV
try:
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
    
    # Mensaje de finalización
    print(f"Datos de OpenInsider guardados en: {file_path} (total: {len(all_rows)} filas)")

except Exception as e:
    print(f"Error al escribir el archivo CSV: {e}")