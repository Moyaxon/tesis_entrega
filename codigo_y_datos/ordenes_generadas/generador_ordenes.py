from pykml import parser
import os
from random import randint, shuffle
from orden_aleatoria import generar_orden_aleatoria_por_tipos as generador

def leer_coordenadas_kml(numero_por_tipo, limite_suma_tipos):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kml_file_path = os.path.join(current_dir, 'coordenadas.kml')
    placemarks = []
    try:
        # Lee el archivo KML
        with open(kml_file_path, 'r', encoding='utf-8') as f:
            root = parser.parse(f).getroot()
        # Busca todos los elementos Placemark en la estructura
        for placemark in root.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
            
            name = placemark.name.text if hasattr(placemark, 'name') else 'Sin nombre'
            
            coordinates = []
            for point in placemark.Point.coordinates.text.strip().split():
                coords = point.split(',')
                if len(coords) >= 2:
                    lon, lat = map(float, coords[:2])
                    coordinates.append((lat, lon))
            
            placemarks.append({
                'Nombre': name,
                'Coordenadas': coordinates
            })
        
        # Extrae las coordenadas de todos los placemarks
        todas_coordenadas = [coordenada for placo in placemarks for coordenada in placo['Coordenadas']]
        
        # Escribe las coordenadas en archivos TXT con nombres incrementales
        base_name = 'customer_order'
        valores_totales = []
        files = 0
        while files < numero_por_tipo*limite_suma_tipos:
            valores_largo_peso =[]
            counter = 0
            while True:
                output_file_path = os.path.join(current_dir, f"{base_name}_numero_{numero_por_tipo}_limite_{limite_suma_tipos}.txt")
                with open(output_file_path, 'w') as f:
                    for coordenada in todas_coordenadas:
                        if counter == 0:
                            f.write(f"{counter} {coordenada[0]} {coordenada[1]} - - -\n")
                        else:
                            orden = generador(numero_por_tipo, limite_suma_tipos)
                            valores_largo_peso.append(orden[0])
                            if coordenada == todas_coordenadas[-1]:
                                f.write(f"D{counter} {coordenada[0]} {coordenada[1]} {randint(10,20)} {randint(30,40)} {orden[1]}")
                            else:
                                f.write(f"D{counter} {coordenada[0]} {coordenada[1]} {randint(10,20)} {randint(30,40)} {orden[1]}\n")
                        counter +=1
                print(f"Coordenadas guardadas en {output_file_path}")
                # Verifica si se escribió algo en el archivo
                if os.path.getsize(output_file_path) > 0:
                    break
                counter += 1
            shuffle(todas_coordenadas)
            files += 1
            valores_totales.append(valores_largo_peso)
    
    except FileNotFoundError:
        print("El archivo KML no se encontró en la ubicación esperada.")
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo KML o procesar los datos: {str(e)}")
# Prueba de la función
numero_maximo_por_tipo = [1,2,3,4,5]
limite_maximo_suma_tipos = [5,10,15,20,25]
for numero in numero_maximo_por_tipo:
    for limite in limite_maximo_suma_tipos:
        leer_coordenadas_kml(numero, limite)

