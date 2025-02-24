import os
import re
import pandas as pd

def lectura(arh1,arh2):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resultado = os.path.join(current_dir, arh1)
    f = open(resultado, "r")
    datos_arch1 = f.readlines()
    f.close()
    resultado = os.path.join(current_dir, arh2)
    f = open(resultado, "r")
    datos_arch2 = f.readlines()
    f.close()
    datos = []
    for linea in range(len(datos_arch1)):
        if linea == 0:
            continue
        l = datos_arch1[linea].replace("\n","").split()
        datos.append([l[0],l[1],l[2],l[3]])
    for linea in range(len(datos_arch2)):
        if linea == 0:
            continue
        l = datos_arch2[linea].replace("\n","").split()
        datos[linea-1].append(l[3])
    
    archivo_analisis = os.path.join(current_dir, f"analisis_tiempos_juntos.txt")
    f = open(archivo_analisis, "w")
    f.write("n_clientes,maximo_por_tipo,maximo_por_cliente,tiempo_HVRP-FVL_prueba_1,tiempo_HVRP-FVL_prueba_2\n")
    for i in range(len(datos)):
        if datos[i] == datos[-1]:
            f.write(f"{datos[i][0]},{datos[i][1]},{datos[i][2]},{datos[i][3]},{datos[i][4]}")
            continue
        f.write(f"{datos[i][0]},{datos[i][1]},{datos[i][2]},{datos[i][3]},{datos[i][4]}\n")
    f.close()

    df = pd.read_csv(archivo_analisis, delimiter=',')
    excel = os.path.join(current_dir, 'analisis_tiempos.xlsx')
    df.to_excel(excel, index=False)
    print("Data successfully converted and saved to analisis_tiempos.xlsx")
    #return info

felipe = "resultados_y_tiempos_felipe.txt"
bastian = "resultados_y_tiempos_bastian.txt"

analisis = lectura(felipe,bastian)